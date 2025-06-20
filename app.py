import streamlit as st
import pickle
import numpy as np
import os

# Load the scaler and model with error handling
@st.cache_resource
def load_models():
    try:
        scaler_path = r"C:\Users\ragul\OneDrive\Desktop\CareerPath_Recommender-main\Jupiter file & dataset\scaler.pkl"
        model_path = r"C:\Users\ragul\OneDrive\Desktop\CareerPath_Recommender-main\Jupiter file & dataset\model.pkl"
        
        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            st.error("Model files not found. Please check the file paths.")
            return None, None
            
        scaler = pickle.load(open(scaler_path, 'rb'))
        model = pickle.load(open(model_path, 'rb'))
        return scaler, model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Updated class names without 'Unknown' and more relevant careers
class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Software Engineer', 
               'Teacher', 'Business Owner', 'Scientist', 'Banker', 'Writer', 
               'Accountant', 'Designer', 'Construction Engineer', 'Game Developer', 
               'Stock Investor', 'Real Estate Developer', 'Data Scientist', 'Pharmacist']

# Career descriptions for better understanding
career_descriptions = {
    'Lawyer': 'Legal practice, litigation, corporate law, human rights law',
    'Doctor': 'Medicine, surgery, specialized medical practice, healthcare',
    'Government Officer': 'Civil services, public administration, policy making',
    'Artist': 'Fine arts, digital art, illustration, creative expression',
    'Software Engineer': 'Programming, software development, web development, mobile apps',
    'Teacher': 'Education, academic instruction, curriculum development',
    'Business Owner': 'Entrepreneurship, business management, startup ventures',
    'Scientist': 'Research, laboratory work, scientific analysis, innovation',
    'Banker': 'Financial services, investment banking, risk management',
    'Writer': 'Content creation, journalism, copywriting, publishing',
    'Accountant': 'Financial accounting, auditing, tax planning, bookkeeping',
    'Designer': 'Graphic design, UI/UX design, product design, branding',
    'Construction Engineer': 'Civil engineering, project management, infrastructure',
    'Game Developer': 'Game programming, game design, interactive media',
    'Stock Investor': 'Financial analysis, portfolio management, trading',
    'Real Estate Developer': 'Property development, real estate investment, construction',
    'Data Scientist': 'Data analysis, machine learning, statistics, AI',
    'Pharmacist': 'Pharmaceutical sciences, drug dispensing, healthcare'
}

# Enhanced recommendation function with better logic
def Recommendations(gender, part_time_job, extracurricular_activities,
                    weekly_self_study_hours, tamil_score, english_score, math_score,
                    bio_cs_score, physics_score, chemistry_score, total_score, average_score):
    
    scaler, model = load_models()
    if scaler is None or model is None:
        return []
    
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0

    # Create feature array matching your original model's expected input format
    # Assuming your model expects: gender, part_time_job, extracurricular, study_hours, 
    # math, history, physics, chemistry, biology, english, geography, total, average
    # We'll map our 6 subjects to the 7 expected subjects
    feature_array = np.array([[gender_encoded, part_time_job_encoded, extracurricular_activities_encoded,
                               weekly_self_study_hours, math_score, tamil_score, physics_score,
                               chemistry_score, bio_cs_score, english_score, 
                               (tamil_score + english_score) / 2,  # geography approximation
                               total_score, average_score]])

    try:
        # Scale features
        scaled_features = scaler.transform(feature_array)

        # Predict using the model
        probabilities = model.predict_proba(scaled_features)

        # Get top five predicted classes along with their probabilities
        top_classes_idx = np.argsort(-probabilities[0])[:5]
        
        # Filter out low probability predictions and create recommendations
        recommendations = []
        for idx in top_classes_idx:
            prob = probabilities[0][idx]
            if prob > 0.05:  # Only show recommendations with >5% probability
                career_name = class_names[idx] if idx < len(class_names) else "Career Option"
                recommendations.append((career_name, prob))
        
        return recommendations
    
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return []

# Streamlit UI setup
def main():
    # Set the page layout to wide
    st.set_page_config(page_title="🎓 Career Guidance System", page_icon="🎓", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .score-display {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .subject-input {
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎓 Career Guidance Recommendation System</h1>', unsafe_allow_html=True)
    
    st.write("""
    ### Welcome to Your Career Path Explorer! 
    This intelligent system analyzes your academic performance across 6 key subjects (Total: 600 marks) 
    and provides AI-powered career recommendations tailored to your strengths and interests.
    """)
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📋 Personal Information")
        
        # Personal details with improved defaults
        gender = st.selectbox("👤 Gender", ["Male", "Female"], index=0, key="gender")
        part_time_job = st.selectbox("💼 Part-Time Job", ["Yes", "No"], index=1, key="part_time_job")
        extracurricular_activities = st.selectbox("🎭 Extracurricular Activities", ["Yes", "No"], index=0, key="extracurricular_activities")
        weekly_self_study_hours = st.number_input("⏱️ Weekly Self-Study Hours", min_value=0, max_value=100, step=1, value=10, key="weekly_study_hours")
        
        st.header("📊 Subject Scores (Out of 100 each - Total 600 marks)")
        st.write("*Enter your scores for each subject:*")
        
        # Subject scores organized in two columns
        col_sub1, col_sub2 = st.columns(2)
        
        with col_sub1:
            tamil_score = st.number_input("🔤 Tamil", min_value=0, max_value=100, value=75, step=1, key="tamil_score")
            english_score = st.number_input("📚 English", min_value=0, max_value=100, value=75, step=1, key="english_score")
            math_score = st.number_input("🔢 Mathematics", min_value=0, max_value=100, value=75, step=1, key="math_score")
        
        with col_sub2:
            bio_cs_score = st.number_input("🧬/💻 Biology/Computer Science", min_value=0, max_value=100, value=75, step=1, key="bio_cs_score")
            physics_score = st.number_input("⚛️ Physics", min_value=0, max_value=100, value=75, step=1, key="physics_score")
            chemistry_score = st.number_input("⚗️ Chemistry", min_value=0, max_value=100, value=75, step=1, key="chemistry_score")
    
    with col2:
        # Real-time score calculation
        total_score = tamil_score + english_score + math_score + bio_cs_score + physics_score + chemistry_score
        average_score = total_score / 6
        percentage = (total_score / 600) * 100
        
        st.markdown(f"""
        <div class="score-display">
        <h3>📈 Your Academic Summary</h3>
        <p><strong>Total Score:</strong> {total_score}/600</p>
        <p><strong>Average Score:</strong> {average_score:.1f}/100</p>
        <p><strong>Overall Percentage:</strong> {percentage:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Grade calculation with Indian grading system
        if percentage >= 90:
            grade = "A+ (Outstanding)"
            color = "#4CAF50"
        elif percentage >= 80:
            grade = "A (Excellent)"
            color = "#8BC34A"
        elif percentage >= 70:
            grade = "B+ (Very Good)"
            color = "#FFC107"
        elif percentage >= 60:
            grade = "B (Good)"
            color = "#FF9800"
        elif percentage >= 50:
            grade = "C (Average)"
            color = "#FF5722"
        elif percentage >= 35:
            grade = "D (Pass)"
            color = "#795548"
        else:
            grade = "F (Fail)"
            color = "#F44336"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                    color: white; padding: 1.5rem; border-radius: 10px; text-align: center; 
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h3>Grade: {grade}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Subject strength analysis
        subjects = {
            'Tamil': tamil_score,
            'English': english_score, 
            'Maths': math_score,
            'Bio/CS': bio_cs_score,
            'Physics': physics_score,
            'Chemistry': chemistry_score
        }
        
        strongest_subject = max(subjects, key=subjects.get)
        st.info(f"💪 **Strongest Subject:** {strongest_subject} ({subjects[strongest_subject]}%)")
    
    # Get recommendations button
    if st.button("🔍 Get AI-Powered Career Recommendations", type="primary"):
        if average_score < 35:
            st.error("⚠️ Your overall performance is below passing grade. Focus on improving your scores to at least 35% average before exploring career paths.")
        else:
            with st.spinner("🤖 AI is analyzing your profile..."):
                # Get recommendations
                recommendations = Recommendations(gender, part_time_job == "Yes", extracurricular_activities == "Yes",
                                                weekly_self_study_hours, tamil_score, english_score, math_score,
                                                bio_cs_score, physics_score, chemistry_score, total_score, average_score)
                
                if recommendations:
                    st.success("✅ Analysis Complete! Here are your personalized recommendations:")
                    st.markdown("---")
                    st.header("🎯 Your AI-Powered Career Recommendations")
                    
                    for i, (career, probability) in enumerate(recommendations, 1):
                        confidence_percentage = probability * 100
                        
                        # Get career description
                        description = career_descriptions.get(career, "Explore this exciting career path")
                        
                        # Color coding based on confidence
                        if confidence_percentage >= 70:
                            confidence_color = "#4CAF50"
                            confidence_text = "High Match"
                        elif confidence_percentage >= 50:
                            confidence_color = "#FF9800"
                            confidence_text = "Good Match"
                        else:
                            confidence_color = "#2196F3"
                            confidence_text = "Potential Match"
                        
                        st.markdown(f"""
                        <div class="recommendation-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3>#{i} {career}</h3>
                            <div style="background-color: {confidence_color}; padding: 0.5rem 1rem; 
                                        border-radius: 20px; font-weight: bold;">
                                {confidence_text} ({confidence_percentage:.1f}%)
                            </div>
                        </div>
                        <p style="margin-top: 1rem; font-size: 1.1rem;">
                            <strong>Career Paths:</strong> {description}
                        </p>
                        <p style="margin-top: 0.5rem; opacity: 0.9;">
                            <strong>AI Confidence:</strong> This career aligns well with your academic profile and interests.
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Additional guidance section
                    st.markdown("---")
                    st.header("💡 Next Steps & Guidance")
                    
                    col_guide1, col_guide2 = st.columns(2)
                    
                    with col_guide1:
                        st.info("""
                        **🎯 Immediate Actions:**
                        1. Research your top 3 recommended careers
                        2. Connect with professionals in these fields
                        3. Look for internships or job shadowing opportunities
                        4. Consider relevant skill development courses
                        """)
                    
                    with col_guide2:
                        st.success("""
                        **🚀 Long-term Planning:**
                        1. Choose college courses aligned with your career goals
                        2. Build a portfolio of relevant projects
                        3. Join professional associations in your field
                        4. Stay updated with industry trends
                        """)
                    
                    # Subject improvement suggestions
                    if average_score < 75:
                        weak_subjects = [subject for subject, score in subjects.items() if score < 60]
                        if weak_subjects:
                            st.warning(f"📚 **Improvement Tip:** Focus on strengthening {', '.join(weak_subjects)} to unlock more career opportunities.")
                
                else:
                    st.error("Unable to generate recommendations. Please check if the model files are properly loaded.")

if __name__ == '__main__':
    main()