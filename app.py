"""
Semantic Cache QnA Chatbot — Streamlit app
Three chat modes: direct LLM | Redis SemanticCache | In-memory FuzzyCache.
"""

import hashlib
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import redis
import streamlit as st
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

# GPT OSS and Llama family models on Groq (prices per 1K tokens)
# Source: https://console.groq.com/docs/models
GROQ_MODELS = {
    "llama-3.3-70b-versatile":                  {"input": 0.00059,  "output": 0.00079},  # $0.59/$0.79 per 1M
    "openai/gpt-oss-120b":                       {"input": 0.00015,  "output": 0.00060},  # $0.15/$0.60 per 1M
    "openai/gpt-oss-20b":                        {"input": 0.000075, "output": 0.00030},  # $0.075/$0.30 per 1M
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.00011,  "output": 0.00034},  # $0.11/$0.34 per 1M
}

# FAQ seed data, loaded from data/faq_seed.csv
FAQ_SEED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "faq_seed.csv")
FAQ_SEED = list(
    pd.read_csv(FAQ_SEED_PATH)[["question", "answer"]].itertuples(index=False, name=None)
)

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. "
    "Answer the customer question concisely and professionally in 1-3 sentences. "
    "If you don't have specific information, give a general helpful response."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))

def calc_cost(model: str, input_text: str, output_text: str) -> float:
    rates = GROQ_MODELS.get(model, {"input": 0.00059, "output": 0.00079})
    return (
        (count_tokens(input_text) / 1000) * rates["input"]
        + (count_tokens(output_text) / 1000) * rates["output"]
    )

# ── FuzzyCache ───────────────

class FuzzyCache:
    """In-memory cache using fuzzywuzzy character-level ratio matching."""

    def __init__(self):
        self._store: list[tuple[str, str]] = []

    def hydrate(self, pairs: list[tuple[str, str]], clear: bool = True) -> None:
        if clear:
            self._store = []
        self._store.extend(pairs)

    def check(self, query: str, distance_threshold: float = 0.4) -> Optional[dict]:
        """
        Returns best-match dict (prompt, response, distance, ratio) if within
        threshold, else None.  distance = 1 - fuzz.ratio/100  (lower = better).
        """
        if not self._store:
            return None
        best_ratio, best_pair = 0, None
        for q, a in self._store:
            r = fuzz.ratio(query.lower(), q.lower())
            if r > best_ratio:
                best_ratio, best_pair = r, (q, a)
        distance = 1 - best_ratio / 100.0
        if distance > distance_threshold:
            return None
        return {
            "prompt":    best_pair[0],
            "response":  best_pair[1],
            "distance":  round(distance, 4),
            "ratio":     best_ratio,
        }

    def store(self, question: str, answer: str) -> None:
        self._store.append((question, answer))


# ── Session-state resource initialisation ────────────

def get_llm(api_key: str, model: str) -> ChatGroq:
    key_hash = hashlib.md5(f"{api_key}{model}".encode()).hexdigest()[:10]
    if st.session_state.get("_llm_hash") != key_hash:
        st.session_state._llm = ChatGroq(
            model=model, temperature=0.1, max_tokens=250, api_key=api_key
        )
        st.session_state._llm_hash = key_hash
    return st.session_state._llm

def get_sem_cache(redis_url: str, distance_threshold: float) -> SemanticCache:
    cfg_hash = hashlib.md5(f"{redis_url}{distance_threshold}".encode()).hexdigest()[:10]
    if st.session_state.get("_cache_hash") != cfg_hash:
        r = redis.Redis.from_url(redis_url)
        r.ping()
        vectorizer = HFTextVectorizer(
            model="sentence-transformers/all-MiniLM-L6-v2",
            cache=EmbeddingsCache(redis_client=r, ttl=3600),
        )
        cache = SemanticCache(
            name="chatbot-faq-cache",
            vectorizer=vectorizer,
            redis_client=r,
            distance_threshold=distance_threshold,
        )
        cache.set_ttl(86400)
        for question, answer in FAQ_SEED:
            cache.store(prompt=question, response=answer)
        st.session_state._sem_cache = cache
        st.session_state._cache_hash = cfg_hash
    return st.session_state._sem_cache

def get_fuzzy_cache() -> FuzzyCache:
    if "_fuzzy_cache" not in st.session_state:
        fc = FuzzyCache()
        fc.hydrate(FAQ_SEED)
        st.session_state._fuzzy_cache = fc
    return st.session_state._fuzzy_cache

# ── Session-state data init ───────────────────────────────────────────────────

def _blank_nc_stats() -> dict:
    return {"total": 0, "latencies": [], "cost": 0.0, "log": []}

def _blank_cache_stats() -> dict:
    return {
        "total": 0,
        "hits": 0,
        "misses": 0,
        "hit_latencies": [],
        "miss_latencies": [],
        "cost": 0.0,       # actual cost paid (misses only)
        "saved_cost": 0.0, # hypothetical LLM cost avoided by hits
        "log": [],
    }

def _blank_fuzzy_stats() -> dict:
    return {
        "total": 0,
        "hits": 0,
        "misses": 0,
        "hit_latencies": [],
        "miss_latencies": [],
        "cost": 0.0,
        "saved_cost": 0.0,
        "log": [],
    }

def _blank_integrated_stats() -> dict:
    return {
        "total": 0,
        "semantic_hits": 0,
        "fuzzy_hits": 0,
        "llm_calls": 0,
        "semantic_latencies": [],   # latency when served by semantic cache
        "fuzzy_latencies": [],      # latency when served by fuzzy cache
        "llm_latencies": [],        # latency when LLM was called
        "cost": 0.0,
        "saved_by_semantic": 0.0,
        "saved_by_fuzzy": 0.0,
        "log": [],
    }

def init_session_state() -> None:
    defaults = {
        "msgs_nc":          [],
        "msgs_cache":       [],
        "msgs_fuzzy":       [],
        "msgs_integrated":  [],
        "stats_nc":         _blank_nc_stats(),
        "stats_cache":      _blank_cache_stats(),
        "stats_fuzzy":      _blank_fuzzy_stats(),
        "stats_integrated": _blank_integrated_stats(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Semantic Cache Chatbot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None,
    },
)

st.title("⚡ Semantic Cache QnA Chatbot")
st.caption(
    "Compare **direct LLM calls** vs **Redis Semantic Cache** — "
    "see latency and cost savings in real time."
)

init_session_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    groq_api_key = st.text_input(
        "Groq API Key", type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="console.groq.com",
    )
    model_choice = st.selectbox("Model", list(GROQ_MODELS.keys()), index=0)
    redis_url = st.text_input("Redis URL", value="redis://localhost:6379")
    distance_threshold = st.slider(
        "Semantic Cache Threshold", 0.05, 0.80, 0.30, 0.05,
        help="Cosine distance cutoff for Redis semantic cache.",
    )
    fuzzy_threshold = st.slider(
        "Fuzzy Match Threshold", 0.10, 0.90, 0.40, 0.05,
        help="1 − fuzz.ratio/100 cutoff. Lower = stricter character match.",
    )

    st.divider()

    if st.button("🗑️ Clear All Chats & Stats", use_container_width=True):
        resets = {
            "stats_nc":         _blank_nc_stats(),
            "stats_cache":      _blank_cache_stats(),
            "stats_fuzzy":      _blank_fuzzy_stats(),
            "stats_integrated": _blank_integrated_stats(),
            "msgs_nc": [], "msgs_cache": [], "msgs_fuzzy": [], "msgs_integrated": [],
        }
        for k, v in resets.items():
            st.session_state[k] = v
        st.rerun()

    if st.button("🔄 Reset Redis Cache", use_container_width=True):
        existing = st.session_state.pop("_sem_cache", None)
        if existing is not None:
            try:
                existing.clear()
            except Exception:
                pass
        st.session_state.pop("_cache_hash", None)
        st.rerun()

    if st.button("🔄 Reset Fuzzy Cache", use_container_width=True):
        st.session_state.pop("_fuzzy_cache", None)
        st.rerun()

    st.divider()
    rates = GROQ_MODELS[model_choice]
    st.markdown(
        f"**{model_choice}**  \n"
        f"`${rates['input']*1000:.3f}` / 1M in  \n"
        f"`${rates['output']*1000:.3f}` / 1M out"
    )
    st.divider()
    st.markdown("**Pre-loaded FAQ:**")
    for q, _ in FAQ_SEED:
        st.caption(f"• {q}")

# ── Validate & init resources ─────────────────────────────────────────────────

if not groq_api_key:
    st.warning("Enter your **Groq API key** in the sidebar to get started.")
    st.stop()

try:
    llm = get_llm(groq_api_key, model_choice)
except Exception as exc:
    st.error(f"LLM init error: {exc}")
    st.stop()

try:
    sem_cache = get_sem_cache(redis_url, distance_threshold)
except redis.ConnectionError:
    st.error(f"Cannot connect to Redis at `{redis_url}`. Run `redis-server` first.")
    st.stop()
except Exception as exc:
    st.error(f"Cache init error: {exc}")
    st.stop()

fuzzy_cache = get_fuzzy_cache()

# ── Shared input-form renderer ────────────────────────────────────────────────

def chat_input_form(form_key: str) -> str:
    """Top-pinned input form. Returns stripped query on submit, else empty string."""
    with st.form(form_key, clear_on_submit=True, border=False):
        c_inp, c_btn = st.columns([8, 1])
        with c_inp:
            text = st.text_input(
                "q", placeholder="Ask a customer support question…",
                label_visibility="collapsed",
            )
        with c_btn:
            sent = st.form_submit_button("Send ➤", use_container_width=True, type="primary")
    return text.strip() if (sent and text.strip()) else ""

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_nc, tab_cache, tab_fuzzy, tab_integrated, tab_dash = st.tabs([
    "💬 Without Cache", "⚡ Semantic Cache", "🔤 Fuzzy Match",
    "🔗 Integrated", "📊 Dashboard"
])

# ── Without Cache tab ─────────────────────────────────────────────────────────

with tab_nc:
    query_nc = chat_input_form("form_nc")
    st.divider()

    for msg in st.session_state.msgs_nc:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("badge"):
                st.caption(msg["badge"])

    if query_nc:
        with st.chat_message("user"):
            st.markdown(query_nc)
        st.session_state.msgs_nc.append({"role": "user", "content": query_nc})

        full_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {query_nc}"

        with st.spinner("Calling Groq…"):
            t0 = time.perf_counter()
            resp = llm.invoke([HumanMessage(content=full_prompt)])
            latency = time.perf_counter() - t0

        response = resp.content.strip()
        cost = calc_cost(model_choice, full_prompt, response)
        badge = f"🤖 **LLM Call** · {latency * 1000:.1f} ms · cost: ${cost:.6f}"

        s = st.session_state.stats_nc
        s["total"] += 1
        s["latencies"].append(latency)
        s["cost"] += cost
        s["log"].append({
            "Query":        query_nc[:55] + ("…" if len(query_nc) > 55 else ""),
            "Latency (ms)": f"{latency * 1000:.1f}",
            "Cost ($)":     f"{cost:.6f}",
        })

        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(badge)
        st.session_state.msgs_nc.append({"role": "assistant", "content": response, "badge": badge})
        st.rerun()

# ── With Cache tab ────────────────────────────────────────────────────────────

with tab_cache:
    query_c = chat_input_form("form_cache")
    st.divider()

    for msg in st.session_state.msgs_cache:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("badge"):
                st.caption(msg["badge"])

    if query_c:
        with st.chat_message("user"):
            st.markdown(query_c)
        st.session_state.msgs_cache.append({"role": "user", "content": query_c})

        full_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {query_c}"
        s = st.session_state.stats_cache

        with st.spinner("Checking Redis cache…"):
            t0 = time.perf_counter()
            cache_result = sem_cache.check(query_c)
            check_latency = time.perf_counter() - t0

        if cache_result:
            latency    = check_latency
            response   = cache_result[0]["response"]
            distance   = cache_result[0]["vector_distance"]
            matched_q  = cache_result[0]["prompt"]
            saved      = calc_cost(model_choice, full_prompt, response)
            actual_cost = 0.0

            badge = (
                f"⚡ **Cache HIT** · {latency * 1000:.1f} ms · "
                f"matched: *\"{matched_q[:55]}\"* · distance: {distance:.3f}"
            )
            s["hits"] += 1
            s["hit_latencies"].append(latency)
            s["saved_cost"] += saved
            s["log"].append({
                "Query":        query_c[:55] + ("…" if len(query_c) > 55 else ""),
                "Result":       "✅ HIT",
                "Latency (ms)": f"{latency * 1000:.1f}",
                "Cost ($)":     "0.000000",
                "Saved ($)":    f"{saved:.6f}",
            })
        else:
            with st.spinner("Cache miss — calling Groq…"):
                t1 = time.perf_counter()
                llm_resp = llm.invoke([HumanMessage(content=full_prompt)])
                latency = check_latency + (time.perf_counter() - t1)

            response    = llm_resp.content.strip()
            actual_cost = calc_cost(model_choice, full_prompt, response)
            sem_cache.store(prompt=query_c, response=response)

            badge = (
                f"🤖 **LLM Call** · {latency * 1000:.1f} ms · "
                f"cost: ${actual_cost:.6f} · stored in cache"
            )
            s["misses"] += 1
            s["miss_latencies"].append(latency)
            s["log"].append({
                "Query":        query_c[:55] + ("…" if len(query_c) > 55 else ""),
                "Result":       "❌ MISS",
                "Latency (ms)": f"{latency * 1000:.1f}",
                "Cost ($)":     f"{actual_cost:.6f}",
                "Saved ($)":    "0.000000",
            })

        s["total"] += 1
        s["cost"] += actual_cost

        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(badge)
        st.session_state.msgs_cache.append({"role": "assistant", "content": response, "badge": badge})
        st.rerun()

# ── Fuzzy Match tab ───────────────────────────────────────────────────────────

with tab_fuzzy:
    query_f = chat_input_form("form_fuzzy")
    st.divider()

    for msg in st.session_state.msgs_fuzzy:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("badge"):
                st.caption(msg["badge"])

    if query_f:
        with st.chat_message("user"):
            st.markdown(query_f)
        st.session_state.msgs_fuzzy.append({"role": "user", "content": query_f})

        full_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {query_f}"
        sf = st.session_state.stats_fuzzy

        t0 = time.perf_counter()
        fuzzy_result = fuzzy_cache.check(query_f, distance_threshold=fuzzy_threshold)
        check_latency = time.perf_counter() - t0

        if fuzzy_result:
            latency     = check_latency
            response    = fuzzy_result["response"]
            matched_q   = fuzzy_result["prompt"]
            ratio       = fuzzy_result["ratio"]
            distance    = fuzzy_result["distance"]
            saved       = calc_cost(model_choice, full_prompt, response)
            actual_cost = 0.0

            badge = (
                f"🔤 **Fuzzy HIT** · {latency * 1000:.1f} ms · "
                f"matched: *\"{matched_q[:55]}\"* · ratio: {ratio}% · distance: {distance:.3f}"
            )
            sf["hits"] += 1
            sf["hit_latencies"].append(latency)
            sf["saved_cost"] += saved
            sf["log"].append({
                "Query":        query_f[:55] + ("…" if len(query_f) > 55 else ""),
                "Result":       "✅ HIT",
                "Ratio (%)":    str(ratio),
                "Latency (ms)": f"{latency * 1000:.1f}",
                "Cost ($)":     "0.000000",
                "Saved ($)":    f"{saved:.6f}",
            })
        else:
            with st.spinner("No fuzzy match — calling Groq…"):
                t1 = time.perf_counter()
                llm_resp = llm.invoke([HumanMessage(content=full_prompt)])
                latency = check_latency + (time.perf_counter() - t1)

            response    = llm_resp.content.strip()
            actual_cost = calc_cost(model_choice, full_prompt, response)
            fuzzy_cache.store(query_f, response)

            badge = (
                f"🤖 **LLM Call** · {latency * 1000:.1f} ms · "
                f"cost: ${actual_cost:.6f} · stored in fuzzy cache"
            )
            sf["misses"] += 1
            sf["miss_latencies"].append(latency)
            sf["log"].append({
                "Query":        query_f[:55] + ("…" if len(query_f) > 55 else ""),
                "Result":       "❌ MISS",
                "Ratio (%)":    "—",
                "Latency (ms)": f"{latency * 1000:.1f}",
                "Cost ($)":     f"{actual_cost:.6f}",
                "Saved ($)":    "0.000000",
            })

        sf["total"] += 1
        sf["cost"] += actual_cost

        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(badge)
        st.session_state.msgs_fuzzy.append({"role": "assistant", "content": response, "badge": badge})
        st.rerun()

# ── Integrated tab ────────────────────────────────────────────────────────────

with tab_integrated:
    query_i = chat_input_form("form_integrated")

    # Live mini-stats bar
    si = st.session_state.stats_integrated
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total",          si["total"])
    m2.metric("⚡ Semantic HITs", si["semantic_hits"])
    m3.metric("🔤 Fuzzy HITs",   si["fuzzy_hits"])
    m4.metric("🤖 LLM Calls",   si["llm_calls"])

    st.divider()

    for msg in st.session_state.msgs_integrated:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("badge"):
                st.caption(msg["badge"])

    if query_i:
        with st.chat_message("user"):
            st.markdown(query_i)
        st.session_state.msgs_integrated.append({"role": "user", "content": query_i})

        full_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {query_i}"
        si = st.session_state.stats_integrated

        # ── Step 1: Semantic cache check ──────────────────────────────────────
        with st.spinner("Checking semantic cache…"):
            t0 = time.perf_counter()
            semantic_result = sem_cache.check(query_i)
            semantic_check_ms = time.perf_counter() - t0

        if semantic_result:
            latency     = semantic_check_ms
            response    = semantic_result[0]["response"]
            distance    = semantic_result[0]["vector_distance"]
            matched_q   = semantic_result[0]["prompt"]
            saved       = calc_cost(model_choice, full_prompt, response)
            actual_cost = 0.0
            source      = "semantic"

            badge = (
                f"⚡ **Semantic HIT** · {latency * 1000:.1f} ms · "
                f"matched: *\"{matched_q[:50]}\"* · distance: {distance:.3f}"
            )
            si["semantic_hits"] += 1
            si["semantic_latencies"].append(latency)
            si["saved_by_semantic"] += saved

        else:
            # ── Step 2: Fuzzy cache check ──────────────────────────────────────
            t1 = time.perf_counter()
            fuzzy_result = fuzzy_cache.check(query_i, distance_threshold=fuzzy_threshold)
            fuzzy_check_ms = time.perf_counter() - t1
            total_check = semantic_check_ms + fuzzy_check_ms

            if fuzzy_result:
                latency     = total_check
                response    = fuzzy_result["response"]
                matched_q   = fuzzy_result["prompt"]
                ratio       = fuzzy_result["ratio"]
                distance    = fuzzy_result["distance"]
                saved       = calc_cost(model_choice, full_prompt, response)
                actual_cost = 0.0
                source      = "fuzzy"

                badge = (
                    f"🔤 **Fuzzy HIT** · {latency * 1000:.1f} ms · "
                    f"matched: *\"{matched_q[:50]}\"* · ratio: {ratio}% · distance: {distance:.3f}"
                )
                si["fuzzy_hits"] += 1
                si["fuzzy_latencies"].append(latency)
                si["saved_by_fuzzy"] += saved

            else:
                # ── Step 3: LLM fallback ───────────────────────────────────────
                with st.spinner("Both caches missed — calling Groq…"):
                    t2 = time.perf_counter()
                    llm_resp = llm.invoke([HumanMessage(content=full_prompt)])
                    latency = total_check + (time.perf_counter() - t2)

                response    = llm_resp.content.strip()
                actual_cost = calc_cost(model_choice, full_prompt, response)
                source      = "llm"

                # Store in BOTH caches for future hits
                sem_cache.store(prompt=query_i, response=response)
                fuzzy_cache.store(query_i, response)

                badge = (
                    f"🤖 **LLM Call** · {latency * 1000:.1f} ms · "
                    f"cost: ${actual_cost:.6f} · stored in semantic + fuzzy cache"
                )
                si["llm_calls"] += 1
                si["llm_latencies"].append(latency)

        saved_amt = saved if source != "llm" else 0.0
        si["total"] += 1
        si["cost"] += actual_cost
        si["log"].append({
            "Query":        query_i[:50] + ("…" if len(query_i) > 50 else ""),
            "Source":       {"semantic": "⚡ Semantic", "fuzzy": "🔤 Fuzzy", "llm": "🤖 LLM"}[source],
            "Latency (ms)": f"{latency * 1000:.1f}",
            "Cost ($)":     f"{actual_cost:.6f}",
            "Saved ($)":    f"{saved_amt:.6f}",
        })

        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(badge)
        st.session_state.msgs_integrated.append(
            {"role": "assistant", "content": response, "badge": badge}
        )
        st.rerun()

# ── Dashboard tab ─────────────────────────────────────────────────────────────

with tab_dash:
    snc = st.session_state.stats_nc
    sc  = st.session_state.stats_cache
    sf  = st.session_state.stats_fuzzy

    if snc["total"] == 0 and sc["total"] == 0 and sf["total"] == 0:
        st.info("No queries yet — use the chat tabs first.")
        st.stop()

    # ── Derived metrics ───────────────────────────────────────────────────────
    nc_avg_ms    = np.mean(snc["latencies"]) * 1000 if snc["latencies"] else 0.0

    c_hit_ms     = np.mean(sc["hit_latencies"])  * 1000 if sc["hit_latencies"]  else 0.0
    c_miss_ms    = np.mean(sc["miss_latencies"]) * 1000 if sc["miss_latencies"] else 0.0
    sc_hit_rate  = (sc["hits"] / sc["total"] * 100) if sc["total"] > 0 else 0.0
    sc_hypo      = sc["cost"] + sc["saved_cost"]
    sc_savings   = (sc["saved_cost"] / sc_hypo * 100) if sc_hypo > 0 else 0.0
    sc_speedup   = (nc_avg_ms / c_hit_ms) if (c_hit_ms > 0 and nc_avg_ms > 0) else 0.0

    f_hit_ms     = np.mean(sf["hit_latencies"])  * 1000 if sf["hit_latencies"]  else 0.0
    f_miss_ms    = np.mean(sf["miss_latencies"]) * 1000 if sf["miss_latencies"] else 0.0
    fz_hit_rate  = (sf["hits"] / sf["total"] * 100) if sf["total"] > 0 else 0.0
    fz_hypo      = sf["cost"] + sf["saved_cost"]
    fz_savings   = (sf["saved_cost"] / fz_hypo * 100) if fz_hypo > 0 else 0.0
    fz_speedup   = (nc_avg_ms / f_hit_ms) if (f_hit_ms > 0 and nc_avg_ms > 0) else 0.0

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown("#### Semantic Cache")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Queries",      sc["total"])
    k2.metric("Hit Rate",     f"{sc_hit_rate:.1f}%")
    k3.metric("Speedup",      f"{sc_speedup:.1f}×" if sc_speedup > 0 else "—")
    k4.metric("Cost Saved",   f"${sc['saved_cost']:.5f}")
    k5.metric("Savings %",    f"{sc_savings:.1f}%")

    st.markdown("#### Fuzzy Match")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Queries",      sf["total"])
    f2.metric("Hit Rate",     f"{fz_hit_rate:.1f}%")
    f3.metric("Speedup",      f"{fz_speedup:.1f}×" if fz_speedup > 0 else "—")
    f4.metric("Cost Saved",   f"${sf['saved_cost']:.5f}")
    f5.metric("Savings %",    f"{fz_savings:.1f}%")

    st.divider()

    r1l, r1r = st.columns(2)

    # ── Latency bar: all 5 types side by side ─────────────────────────────────
    with r1l:
        bar_labels, bar_vals, bar_colors = [], [], []
        if snc["latencies"]:
            bar_labels.append("No Cache"); bar_vals.append(nc_avg_ms);  bar_colors.append("#e74c3c")
        if sc["hit_latencies"]:
            bar_labels.append("Semantic HIT"); bar_vals.append(c_hit_ms);  bar_colors.append("#2ecc71")
        if sc["miss_latencies"]:
            bar_labels.append("Semantic MISS"); bar_vals.append(c_miss_ms); bar_colors.append("#f39c12")
        if sf["hit_latencies"]:
            bar_labels.append("Fuzzy HIT"); bar_vals.append(f_hit_ms);  bar_colors.append("#3498db")
        if sf["miss_latencies"]:
            bar_labels.append("Fuzzy MISS"); bar_vals.append(f_miss_ms); bar_colors.append("#9b59b6")

        fig_lat = go.Figure(go.Bar(
            x=bar_labels, y=bar_vals, marker_color=bar_colors,
            text=[f"{v:.1f} ms" for v in bar_vals], textposition="outside",
        ))
        fig_lat.update_layout(
            title="Avg Response Latency — All Methods",
            yaxis_title="ms", height=350,
            margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    # ── Grouped pie: hit/miss for semantic + fuzzy ────────────────────────────
    with r1r:
        has_sc = sc["total"] > 0
        has_fz = sf["total"] > 0
        if has_sc or has_fz:
            fig_pie = go.Figure()
            if has_sc:
                fig_pie.add_trace(go.Pie(
                    labels=["Semantic HIT", "Semantic MISS"],
                    values=[sc["hits"], sc["misses"]],
                    marker_colors=["#2ecc71", "#f39c12"],
                    hole=0.45, textinfo="label+percent",
                    name="Semantic", domain={"x": [0, 0.46]},
                ))
            if has_fz:
                fig_pie.add_trace(go.Pie(
                    labels=["Fuzzy HIT", "Fuzzy MISS"],
                    values=[sf["hits"], sf["misses"]],
                    marker_colors=["#3498db", "#9b59b6"],
                    hole=0.45, textinfo="label+percent",
                    name="Fuzzy", domain={"x": [0.54, 1.0]},
                ))
            fig_pie.update_layout(
                title="Hit Rate — Semantic (left) vs Fuzzy (right)",
                height=350, margin=dict(t=50, b=10, l=10, r=10),
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Use **Semantic Cache** or **Fuzzy Match** tabs to see hit rates.")

    st.subheader("💰 Cost Comparison")
    r2l, r2r = st.columns(2)

    # ── Cost bar: 4-column comparison ─────────────────────────────────────────
    with r2l:
        cost_labels = [
            "No Cache",
            "Semantic\n(actual)",
            "Semantic\n(no-cache est.)",
            "Fuzzy\n(actual)",
            "Fuzzy\n(no-cache est.)",
        ]
        cost_vals   = [snc["cost"], sc["cost"], sc_hypo, sf["cost"], fz_hypo]
        cost_colors = ["#e74c3c", "#2ecc71", "#95a5a6", "#3498db", "#bdc3c7"]
        fig_cost = go.Figure(go.Bar(
            x=cost_labels, y=cost_vals, marker_color=cost_colors,
            text=[f"${v:.5f}" for v in cost_vals], textposition="outside",
        ))
        fig_cost.update_layout(
            title="Estimated Cost Comparison",
            yaxis_title="USD", height=350,
            margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    # ── Summary table: semantic vs fuzzy ──────────────────────────────────────
    with r2r:
        st.markdown("#### Summary")
        st.dataframe(pd.DataFrame({
            "Metric": [
                "Queries",
                "Hits", "Misses", "Hit Rate",
                "Avg HIT latency",
                "Avg MISS latency",
                "Speedup vs no-cache",
                "Actual cost",
                "Cost saved",
                "Savings %",
            ],
            "Semantic Cache": [
                str(sc["total"]),
                str(sc["hits"]), str(sc["misses"]), f"{sc_hit_rate:.1f}%",
                f"{c_hit_ms:.1f} ms"  if c_hit_ms  else "—",
                f"{c_miss_ms:.1f} ms" if c_miss_ms else "—",
                f"{sc_speedup:.1f}×"  if sc_speedup > 0 else "—",
                f"${sc['cost']:.5f}",
                f"${sc['saved_cost']:.5f}",
                f"{sc_savings:.1f}%",
            ],
            "Fuzzy Match": [
                str(sf["total"]),
                str(sf["hits"]), str(sf["misses"]), f"{fz_hit_rate:.1f}%",
                f"{f_hit_ms:.1f} ms"  if f_hit_ms  else "—",
                f"{f_miss_ms:.1f} ms" if f_miss_ms else "—",
                f"{fz_speedup:.1f}×"  if fz_speedup > 0 else "—",
                f"${sf['cost']:.5f}",
                f"${sf['saved_cost']:.5f}",
                f"{fz_savings:.1f}%",
            ],
        }), use_container_width=True, hide_index=True)
        st.caption(
            f"Model: `{model_choice}` · "
            f"Semantic threshold: `{distance_threshold}` · "
            f"Fuzzy threshold: `{fuzzy_threshold}`"
        )

    # ── Per-query logs — three columns ────────────────────────────────────────
    if snc["log"] or sc["log"] or sf["log"]:
        st.subheader("📋 Query Logs")
        ll, lm, lr = st.columns(3)
        with ll:
            st.markdown("**Without Cache**")
            if snc["log"]:
                st.dataframe(pd.DataFrame(snc["log"]), use_container_width=True, hide_index=True)
            else:
                st.caption("No queries yet.")
        with lm:
            st.markdown("**Semantic Cache**")
            if sc["log"]:
                st.dataframe(pd.DataFrame(sc["log"]), use_container_width=True, hide_index=True)
            else:
                st.caption("No queries yet.")
        with lr:
            st.markdown("**Fuzzy Match**")
            if sf["log"]:
                st.dataframe(pd.DataFrame(sf["log"]), use_container_width=True, hide_index=True)
            else:
                st.caption("No queries yet.")

    # ── Latency timeline ──────────────────────────────────────────────────────
    total_logged = len(snc["log"]) + len(sc["log"]) + len(sf["log"])
    if total_logged > 1:
        st.subheader("📈 Latency Over Time")
        fig_time = go.Figure()

        if snc["log"]:
            fig_time.add_trace(go.Scatter(
                x=list(range(1, len(snc["log"]) + 1)),
                y=[float(r["Latency (ms)"]) for r in snc["log"]],
                mode="markers+lines", name="No Cache",
                marker=dict(color="#e74c3c", size=8),
                line=dict(color="#e74c3c", width=1, dash="dot"),
            ))
        if sc["log"]:
            hits_y   = [float(r["Latency (ms)"]) if r["Result"] == "✅ HIT"  else None for r in sc["log"]]
            misses_y = [float(r["Latency (ms)"]) if r["Result"] == "❌ MISS" else None for r in sc["log"]]
            xs = list(range(1, len(sc["log"]) + 1))
            fig_time.add_trace(go.Scatter(
                x=xs, y=hits_y, mode="markers", name="Semantic HIT",
                marker=dict(color="#2ecc71", size=8, symbol="circle"),
            ))
            fig_time.add_trace(go.Scatter(
                x=xs, y=misses_y, mode="markers", name="Semantic MISS",
                marker=dict(color="#f39c12", size=8, symbol="x"),
            ))

        if sf["log"]:
            fhits_y  = [float(r["Latency (ms)"]) if r["Result"] == "✅ HIT"  else None for r in sf["log"]]
            fmisses_y= [float(r["Latency (ms)"]) if r["Result"] == "❌ MISS" else None for r in sf["log"]]
            fxs = list(range(1, len(sf["log"]) + 1))
            fig_time.add_trace(go.Scatter(
                x=fxs, y=fhits_y, mode="markers", name="Fuzzy HIT",
                marker=dict(color="#3498db", size=8, symbol="diamond"),
            ))
            fig_time.add_trace(go.Scatter(
                x=fxs, y=fmisses_y, mode="markers", name="Fuzzy MISS",
                marker=dict(color="#9b59b6", size=8, symbol="cross"),
            ))

        fig_time.update_layout(
            xaxis_title="Query #", yaxis_title="Latency (ms)",
            height=300, margin=dict(t=20, b=30, l=10, r=10),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig_time, use_container_width=True)
