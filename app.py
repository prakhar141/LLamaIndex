                    st.toast("Thanks for your feedback!")
            with col2:
                if st.button("👎 Not Helpful"):
                    st.toast("We'll work on it!")

            # Download answer
            st.download_button("📥 Download Answer", answer, file_name="response.txt")

    # Chat History
    with st.expander("🕓 Chat History"):
        for q, a in reversed(st.session_state.history):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")
else:
    st.info("📌 Upload a PDF to get started.")

# ========== Footer ==========
st.markdown(
    """
    <hr style="margin-top: 30px; margin-bottom: 10px;">
    <div style='text-align: center; color: gray; font-size: 14px;'>
        Developed with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani
    </div>
    """,
    unsafe_allow_html=True
)
