import hmac
import json
import os

import streamlit as st

from ui.agreements_ui import render as render_agreements_ui
from ui.generals_ui import render as render_generals_ui
from ui.intents_ui import render as render_intents_ui
from ui.rag_ui import render as render_rag_ui
from utils import load_document_data, load_unified_excels


def check_password():
    expected = os.getenv("APP_PASSWORD")
    if not expected:
        st.error("APP_PASSWORD не задан в окружении.")
        return False

    def password_entered():
        entered = st.session_state.get("password", "")
        ok = hmac.compare_digest(entered, expected)
        st.session_state["password_correct"] = ok
        if ok:
            st.session_state.pop("password", None)

    if not st.session_state.get("password_correct", False):
        st.text_input("Пароль", type="password", key="password", on_change=password_entered)
        st.info(
            "После ввода верного пароля первый запуск может занять некоторое время, пожалуйста, подождите."
        )
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("Неверный пароль")
        return False

    return True


if not check_password():
    st.stop()


st.set_page_config(page_title="Проверка фраз", layout="centered")
st.title("🤖 Проверка фраз")

DOCUMENTS = {
    #"Договорённости": {
       # "urls": [
            #"https://raw.githubusercontent.com/skatzrskx55q/Retrieve2/main/data66.xlsx",
        #],
       # "loader": load_unified_excels,
      #  "loader_kwargs": {
            # Пример точечного override:
            # "parse_profile": {"filter": {"split_newline": False}},
       # },
       # "renderer": render_agreements_ui,
   # },
    "Интенты": {
        "urls": [
            "https://raw.githubusercontent.com/skatzrskx55q/Retrieve2/main/intents22.xlsx",
        ],
        "loader": load_unified_excels,
        "loader_kwargs": {},
        "renderer": render_intents_ui,
    },
    "Generals": {
        "urls": [
            "https://raw.githubusercontent.com/skatzrskx55q/Retrieve2/main/intents33.xlsx",
        ],
        "loader": load_unified_excels,
        "loader_kwargs": {},
        "renderer": render_generals_ui,
    },
  #  "Confluence": {
     #   "urls": [
      #      "https://skatzr.atlassian.net/wiki/spaces/~7120203b1cf4260fea434db9c78c6e8549bd2b/pages/4194305",
     #  ],
     #   "loader": load_document_data,
     #   "loader_kwargs": {},
    #    "renderer": render_rag_ui,
    #},
}

TEAMS = {
   # "Чат-бот": ["Confluence"],
    "Голос": ["Интенты", "Generals"],
    #"Голос": ["Договорённости", "Интенты", "Generals"],
    "Чат-Бот2": [],
    "Чат-Бот3": [],
}

PRELOAD_TEAMS = ("Голос",)


def _loader_kwargs_key(domain_name):
    kwargs = DOCUMENTS[domain_name].get("loader_kwargs") or {}
    return json.dumps(kwargs, sort_keys=True, ensure_ascii=False)


@st.cache_resource(ttl=3600)
def get_data(domain_name, loader_kwargs_key=""):
    _ = loader_kwargs_key  # учитываем конфиг загрузки в ключе кэша
    conf = DOCUMENTS[domain_name]
    loader_kwargs = conf.get("loader_kwargs") or {}
    return conf["loader"](conf["urls"], **loader_kwargs)


def _resolve_preload_docs(team_names):
    ordered_docs = []
    seen = set()
    for team_name in team_names:
        for doc_name in TEAMS.get(team_name, []):
            if doc_name in DOCUMENTS and doc_name not in seen:
                seen.add(doc_name)
                ordered_docs.append(doc_name)
    return ordered_docs


def _preload_voice_docs():
    preload_docs = _resolve_preload_docs(PRELOAD_TEAMS)
    preload_errors = {}

    with st.spinner("Предзагрузка документов. Это может занять 1-2 минуты..."):
        for doc_name in preload_docs:
            try:
                get_data(doc_name, loader_kwargs_key=_loader_kwargs_key(doc_name))
            except Exception as exc:
                preload_errors[doc_name] = str(exc)

    return preload_docs, preload_errors


preload_signature = json.dumps(
    {
        doc_name: DOCUMENTS[doc_name].get("loader_kwargs") or {}
        for doc_name in _resolve_preload_docs(PRELOAD_TEAMS)
    },
    sort_keys=True,
    ensure_ascii=False,
)

if st.session_state.get("preload_signature") != preload_signature:
    preloaded_docs, preload_errors = _preload_voice_docs()
    st.session_state["preloaded_docs"] = preloaded_docs
    st.session_state["preload_errors"] = preload_errors
    st.session_state["preload_signature"] = preload_signature
else:
    preloaded_docs = st.session_state.get("preloaded_docs", [])
    preload_errors = st.session_state.get("preload_errors", {})

with st.sidebar:
    st.header("Выбор команды")
    team = st.radio("Команда", options=list(TEAMS.keys()), index=1)
    team_docs = TEAMS[team]
    st.header("Выбор документа")
    if team_docs:
        domain = st.radio("Документ", options=team_docs, index=0)
    else:
        domain = None
        st.info("Для этой команды документы пока не настроены.")

    if preload_errors:
        st.caption("Некоторые документы не предзагружены. Они будут загружаться при выборе.")


if domain:
    if domain in preload_errors:
        st.warning(
            "Документ не удалось предзагрузить при старте. Пробую загрузить его сейчас."
        )

    try:
        df = get_data(domain, loader_kwargs_key=_loader_kwargs_key(domain))
    except Exception as exc:
        st.error(f"Ошибка загрузки документа «{domain}»: {exc}")
    else:
        DOCUMENTS[domain]["renderer"](df)
