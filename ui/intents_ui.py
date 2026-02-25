import html

import streamlit as st

from utils import group_search_results, keyword_search_rows, semantic_search_rows

GENERAL_FILTER_COL = "display_filter2"
LOCAL_FILTER_COL = "display_filter1"
DESCRIPTION_DISPLAY_COL = "display1"
COMMENT_COL = "comment1"


def render_intent_phrases_html(phrases, best_phrase=None, matched_phrases=None):
    tiles_html = ""
    for phrase in phrases:
        if best_phrase is not None and phrase == best_phrase:
            bg_color = "#ffeb3b"
            font_weight = "bold"
        elif matched_phrases and phrase in matched_phrases:
            bg_color = "#ffeb3b"
            font_weight = "bold"
        else:
            bg_color = "#f9f9f9"
            font_weight = "normal"
        tiles_html += (
            f'<div style="display:inline-block;'
            f'border:1px solid #e0e0e0;'
            f'border-radius:12px;'
            f'padding:8px 12px;'
            f'margin:4px;'
            f'background-color:{bg_color};'
            f'color:#333;'
            f'font-weight:{font_weight};'
            f'font-size:14px;'
            f'box-shadow:0 1px 3px rgba(0,0,0,0.05);">'
            f'{html.escape(phrase)}</div>'
        )
    return tiles_html


def render(df):
    local_col = LOCAL_FILTER_COL
    all_locals = sorted(df[local_col].dropna().astype(str).unique())
    selected_local = st.selectbox(
        "Выберите Local для просмотра (фильтр)",
        options=all_locals,
        index=None,
        placeholder="Выберите Local",
    )

    if selected_local:
        st.subheader(f"Local: {selected_local}")
        local_rows = df[df[local_col] == selected_local].drop_duplicates(subset=["original_index"])
        for _, row in local_rows.iterrows():
            orig_idx = row["original_index"]
            all_examples = df.attrs["original_examples_map"].get(orig_idx, set())
            tiles_html = render_intent_phrases_html(list(all_examples))
            general = str(row.get(GENERAL_FILTER_COL, ""))
            local = str(row.get(LOCAL_FILTER_COL, ""))
            description = str(row.get(DESCRIPTION_DISPLAY_COL, ""))
            comment = str(row.get(COMMENT_COL, ""))

            block_html = f"""
            <div style="border:1px solid #e0e0e0;border-radius:12px;padding:16px;margin-bottom:16px;background-color:#f9f9f9;color:#333;box-shadow:0 2px 6px rgba(0,0,0,0.05);">
                <div style="display:flex; gap:12px; align-items:center; font-size:18px; font-weight:600;">
                    <span style="color:#d32f2f;">🔴 {html.escape(general)}</span>
                    <span style="color:#1976d2;">🔵 {html.escape(local)}</span>
                </div>
                <div style="margin-top:6px;"><strong>Описание General:</strong> {html.escape(description)}</div>
                <div style="margin-top:10px;"><strong>Примеры:</strong></div>
                <div style="margin-top:6px;">{tiles_html}</div>
            </div>
            """
            st.markdown(block_html, unsafe_allow_html=True)

            if comment and comment.strip().lower() != "nan":
                with st.expander("💬 Комментарий", expanded=False):
                    st.markdown(comment)

    query = st.text_input("Введите запрос для поиска по примерам:")
    if query:
        try:
            sem_results_raw = semantic_search_rows(
                query,
                df,
                threshold=0.5,
                filter_cols=[GENERAL_FILTER_COL, LOCAL_FILTER_COL],
                display_cols=[DESCRIPTION_DISPLAY_COL],
                comment_col=COMMENT_COL,
            )
            if sem_results_raw:
                sem_groups = group_search_results(
                    sem_results_raw,
                    df.attrs,
                    search_type="semantic",
                    group_by_filter_cols=[GENERAL_FILTER_COL, LOCAL_FILTER_COL],
                )[:10]
                st.markdown("### 🧠 Результаты умного поиска")
                for group in sem_groups:
                    tiles_html = render_intent_phrases_html(group["all_phrases"], best_phrase=group.get("best_phrase"))
                    score_display = f" (релевантность: {group['max_score']:.2f})" if group.get("max_score") else ""
                    filters = group.get("filters", {})
                    displays = group.get("displays", {})
                    general = filters.get(GENERAL_FILTER_COL, "")
                    local = filters.get(LOCAL_FILTER_COL, "")
                    description = displays.get(DESCRIPTION_DISPLAY_COL, "")
                    block_html = f"""
                    <div style="border:1px solid #e0e0e0;border-radius:12px;padding:16px;margin-bottom:16px;background-color:#f9f9f9;color:#333;box-shadow:0 2px 6px rgba(0,0,0,0.05);">
                        <div style="display:flex; gap:12px; align-items:center; font-size:18px; font-weight:600;">
                            <span style="color:#d32f2f;">🔴 {html.escape(general)}</span>
                            <span style="color:#1976d2;">🔵 {html.escape(local)}</span>
                            {score_display}
                        </div>
                        <div style="margin-top:6px;"><strong>Описание General:</strong> {html.escape(description)}</div>
                        <div style="margin-top:10px;"><strong>Примеры:</strong></div>
                        <div style="margin-top:6px;">{tiles_html}</div>
                    </div>
                    """
                    st.markdown(block_html, unsafe_allow_html=True)
                    if group.get("comment") and str(group["comment"]).strip().lower() != "nan":
                        with st.expander("💬 Комментарий", expanded=False):
                            st.markdown(group["comment"])
            else:
                st.info("Нет результатов семантического поиска.")

            exact_results_raw = keyword_search_rows(
                query,
                df,
                filter_cols=[GENERAL_FILTER_COL, LOCAL_FILTER_COL],
                display_cols=[DESCRIPTION_DISPLAY_COL],
                comment_col=COMMENT_COL,
            )
            if exact_results_raw:
                exact_groups = group_search_results(
                    exact_results_raw,
                    df.attrs,
                    search_type="exact",
                    group_by_filter_cols=[GENERAL_FILTER_COL, LOCAL_FILTER_COL],
                )
                st.markdown("### 📌 Результаты точного поиска")
                for group in exact_groups:
                    tiles_html = render_intent_phrases_html(group["all_phrases"], matched_phrases=group.get("matched_phrases"))
                    filters = group.get("filters", {})
                    displays = group.get("displays", {})
                    general = filters.get(GENERAL_FILTER_COL, "")
                    local = filters.get(LOCAL_FILTER_COL, "")
                    description = displays.get(DESCRIPTION_DISPLAY_COL, "")
                    block_html = f"""
                    <div style="border:1px solid #e0e0e0;border-radius:12px;padding:16px;margin-bottom:16px;background-color:#f9f9f9;color:#333;box-shadow:0 2px 6px rgba(0,0,0,0.05);">
                        <div style="display:flex; gap:12px; align-items:center; font-size:18px; font-weight:600;">
                            <span style="color:#d32f2f;">🔴 {html.escape(general)}</span>
                            <span style="color:#1976d2;">🔵 {html.escape(local)}</span>
                        </div>
                        <div style="margin-top:6px;"><strong>Описание General:</strong> {html.escape(description)}</div>
                        <div style="margin-top:10px;"><strong>Примеры:</strong></div>
                        <div style="margin-top:6px;">{tiles_html}</div>
                    </div>
                    """
                    st.markdown(block_html, unsafe_allow_html=True)
                    if group.get("comment") and str(group["comment"]).strip().lower() != "nan":
                        with st.expander("💬 Комментарий", expanded=False):
                            st.markdown(group["comment"])
            else:
                st.info("Нет результатов точного поиска.")
        except Exception as e:
            st.error(f"Ошибка при обработке запроса: {e}")

