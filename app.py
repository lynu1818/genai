import streamlit as st


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    if st.button("登入"):
        st.session_state.logged_in = True
        st.rerun()

def logout():
    if st.button("登出"):
        st.session_state.logged_in = False
        st.rerun()

login_page = st.Page(login, title="登入", icon=":material/login:")
logout_page = st.Page(logout, title="登出", icon=":material/logout:")

# dashboard = st.Page(
#     "reports/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True
# )
# bugs = st.Page("reports/bugs.py", title="Bug reports", icon=":material/bug_report:")
# alerts = st.Page(
#     "reports/alerts.py", title="System alerts", icon=":material/notification_important:"
# )

search = st.Page("tools/1_.py", title="智能助手", icon=":material/search:")
tool4 = st.Page("tools/4_.py", title="犯罪數據庫", icon=":material/search:")
tool3 = st.Page("tools/3_.py", title="嫌犯人像生成", icon=":material/history:")
history = st.Page("tools/2_.py", title="犯罪現場生成", icon=":material/history:")



if st.session_state.logged_in:
    pg = st.navigation(
        {
            "帳號": [logout_page],
            # "Reports": [dashboard, bugs, alerts],
            "工具": [search, tool4, tool3, history],
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()