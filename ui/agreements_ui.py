import streamlit as st
import torch

from utils import keyword_search_rows, semantic_search_rows


def render_phrases_grid(phrases, cols=3, color="#e0f7fa"):
    rows = [phrases[i:i + cols] for i in range(0, len(phrases), cols)]
    for row in rows:
        cols_streamlit = st.columns(cols)
        for col, phrase in zip(cols_streamlit, row):
            col.markdown(
                f"""<div style="background-color:{color};
                                color:#111;
                                padding:6px 10px;
                                border-radius:12px;
                                display:inline-block;
                                margin:4px;
                                font-size:14px;">{phrase}</div>""",
                unsafe_allow_html=True,
            )


def _topics_for_original(df, original_index):
    rows = df[df["original_index"] == original_index]
    if rows.empty:
        return []

    merged = []
    for topics in rows["topics"].tolist():
        if isinstance(topics, list):
            merged.extend(topics)
    return list(dict.fromkeys(merged))


def render(df):
    all_topics = sorted({topic for topics in df["topics"] for topic in topics})
    tab1, tab2, tab3 = st.tabs(["🔍 Поиск", "🚫 Не используем", "✅/❌ Да и Нет"])

    with tab1:
        selected_topics = st.multiselect(
            "Фильтр по тематикам (независимо от поиска):", all_topics
        )
        filter_search_by_topics = st.checkbox("Искать только в выбранных тематиках", value=False)

        if selected_topics:
            st.markdown("### 📂 Фразы по выбранным тематикам:")
            filtered_df = df[df["topics"].apply(lambda topics: any(t in selected_topics for t in topics))]
            for row in filtered_df.itertuples():
                with st.container():
                    st.markdown(
                        f"""<div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                            <div style="font-size: 18px; font-weight: 600; color: #333;">📝 {row.phrase_full}</div>
                            <div style="margin-top: 4px; font-size: 14px; color: #666;">🔖 Тематики: <strong>{', '.join(row.topics)}</strong></div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    if row.comment1 and str(row.comment1).strip().lower() != "nan":
                        with st.expander("💬 Комментарий", expanded=False):
                            st.markdown(row.comment1)

        query = st.text_input("Введите ваш запрос:")
        if query:
            try:
                search_df = df
                if filter_search_by_topics and selected_topics:
                    mask = df["topics"].apply(lambda topics: any(t in selected_topics for t in topics))
                    search_df = df[mask].copy()
                    if not search_df.empty:
                        idxs = search_df.index.to_list()
                        search_df.attrs["phrase_embs"] = df.attrs["phrase_embs"][idxs]
                    else:
                        search_df.attrs["phrase_embs"] = torch.empty((0, df.attrs["phrase_embs"].shape[1]))

                if search_df.empty:
                    st.warning("Нет данных для поиска по выбранным тематикам.")
                else:
                    results = semantic_search_rows(
                        query,
                        search_df,
                        threshold=0.5,
                        top_k=5,
                        display_cols=["display1"],
                        comment_col="comment1",
                        deduplicate=True,
                    )
                    if results:
                        st.markdown("### 🔍 Результаты умного поиска:")
                        for item in results:
                            score = float(item.get("score", 0.0))
                            phrase_full = item.get("displays", {}).get("display1") or item.get("phrase", "")
                            topics = _topics_for_original(search_df, item.get("original_index"))
                            comment = item.get("comment", "")
                            with st.container():
                                st.markdown(
                                    f"""<div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                                        <div style="font-size: 18px; font-weight: 600; color: #333;">🧠 {phrase_full}</div>
                                        <div style="margin-top: 4px; font-size: 14px; color: #666;">🔖 Тематики: <strong>{', '.join(topics)}</strong></div>
                                        <div style="margin-top: 2px; font-size: 13px; color: #999;">🎯 Релевантность: {score:.2f}</div>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )
                                if comment and str(comment).strip().lower() != "nan":
                                    with st.expander("💬 Комментарий", expanded=False):
                                        st.markdown(comment)
                    else:
                        st.warning("Совпадений не найдено в умном поиске.")

                    exact_results = keyword_search_rows(
                        query,
                        search_df,
                        display_cols=["display1"],
                        comment_col="comment1",
                        deduplicate=True,
                    )
                    if exact_results:
                        st.markdown("### 🧷 Точный поиск:")
                        for item in exact_results:
                            phrase = item.get("displays", {}).get("display1") or item.get("phrase", "")
                            topics = _topics_for_original(search_df, item.get("original_index"))
                            comment = item.get("comment", "")
                            with st.container():
                                st.markdown(
                                    f"""<div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                                        <div style="font-size: 18px; font-weight: 600; color: #333;">📌 {phrase}</div>
                                        <div style="margin-top: 4px; font-size: 14px; color: #666;">🔖 Тематики: <strong>{', '.join(topics)}</strong></div>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )
                                if comment and str(comment).strip().lower() != "nan":
                                    with st.expander("💬 Комментарий", expanded=False):
                                        st.markdown(comment)
                    else:
                        st.info("Ничего не найдено в точном поиске.")
            except Exception as e:
                st.error(f"Ошибка при обработке запроса: {e}")

    with tab2:
        st.markdown("### 🚫 Локалы, которые **не используем**")
        unused_topics = [
            "Local_Balance_Transfer", "Local_Friends", "Local_Next_Payment", "Local_Order_Cash",
            "Local_Other_Cashback", "Local_RemittanceStatus", "Подожди (Wait)", "Local_X5",
            "PassportChangeFirst", "PassportChangeSecond", "Меньше (Local_Less)", "Больше (Local_More)",
            "Рефинансирование под залог недвижимости (Local_Secured_Refinancing)",
            "Действующий займ (Local_Current_MFO_2)",
            "General Мои кредитные предложения (General_My_loan_offers)",
            "Настроить/Изменить/Восстановить (Local_Setup_Secret_Code)",
            "Как сделать устройство доверенным (Local_Trusted_Device)",
            "Что такое доверенное устройство (Local_About_Trusted_Device)",
            "Что такое секретный код (Local_About_Secret_Code)",
            "займы более 100 тыс (Local_MoreNumbers)", "займы меньше 100 тыс (Local_LessNumbers)",
            "Новая карта (NewCard)", "Проблема с начислением кэшбэка (Local_Problem_CashBack)",
        ]
        for topic in unused_topics:
            st.markdown(f"- {topic}")

    with tab3:
        st.markdown("### ✅ Интерпретации 'ДА'")
        yes_phrases = [
            "Подсказать", "Помню", "Хорошо", "Да", "Ага", "Угу", "Да по этому вопросу", "Остались",
            "Можно", "Жги", "Валяй", "Готов", "Ну-ну", "Быстрее", "Проверь", "Проверяй",
            "Все равно хочу", "Подскажите", "Расскажи", "Скажи", "Проверил", "Давал", "Я могу",
            "У меня вопрос есть", "Сказал", "Проконсультируйте", "Пробовала вносите в вашу базу",
        ]
        render_phrases_grid(yes_phrases, cols=3, color="#d1f5d3")
        st.markdown("---")
        st.markdown("### ❌ Интерпретации 'НЕТ'")
        no_phrases = ["Не надо", "Не хочу", "Не готов", "Не помню", "Не пробовала", "Не интересно"]
        render_phrases_grid(no_phrases, cols=3, color="#f9d6d5")



