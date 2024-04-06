import streamlit as st

def main():
    st.set_page_config(layout="wide")

    # Sidebar - Select page
    page = st.sidebar.radio("Navigation", ["Detailed View", "Stock Screener"])

    if page == "Detailed View":
        show_page_1()
    elif page == "Stock Screener":
        show_page_2()


if __name__ == "__main__":
    main()
