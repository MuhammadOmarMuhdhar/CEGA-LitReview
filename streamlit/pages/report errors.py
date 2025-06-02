import streamlit as st
from datetime import datetime

def main():
    st.title("Report Errors")
    
    # Simple form for reporting any type of error
    with st.form("error_report"):
        st.subheader("Found an issue? Let us know!")
        
        # Paper identification
        paper_title = st.text_input("Paper DOI*", placeholder="Enter None if not applicable")
        
        # What's wrong?
        issue_type = st.multiselect(
            "What's the problem?*", 
            ["Wrong labels", "Not about poverty", "Duplicate paper", "Error With The Dashboard", "Other"],
        )
        
        # Simple explanation
        explanation = st.text_area(
            "Tell us more*", 
            placeholder="Describe the issue..."
        )
        
        # Optional contact
        email = st.text_input("Email (optional)", placeholder="your.email@example.com")
        
        submitted = st.form_submit_button("Submit Report")
        
        if submitted:
            if paper_title and explanation:
                # Save report
                report_data = {
                    "timestamp": datetime.now().isoformat(),
                    "paper_title": paper_title,
                    "issue_type": issue_type,
                    "explanation": explanation,
                    "email": email
                }
                
                # Store report (replace with your database)
                if 'reports' not in st.session_state:
                    st.session_state.reports = []
                st.session_state.reports.append(report_data)
                
                st.success("Thanks! We'll review your report.")
            else:
                st.error("Please fill in the required fields (*)")

if __name__ == "__main__":
    main()