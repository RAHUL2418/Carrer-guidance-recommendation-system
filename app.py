import streamlit as st
import pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# For cloud deployment - handle model loading differently
@st.cache_resource
def load_models():
    try:
        # Try to load from cloud deployment paths first
        model_paths = [
            # Local development paths
            r"C:\Users\ragul\OneDrive\Desktop\CareerPath_Recommender-main\Jupiter file & dataset\scaler.pkl",
            r"C:\Users\ragul\OneDrive\Desktop\CareerPath_Recommender-main\Jupiter file & dataset\model.pkl",
            # Cloud deployment paths
            "scaler.pkl",
            "model.pkl",
            "models/scaler.pkl", 
            "models/model.pkl"
        ]
        
        scaler = None
        model = None
        
        # Try different path combinations
        for scaler_path in [p for p in model_paths if 'scaler' in p]:
            if os.path.exists(scaler_path):
                try:
                    scaler = pickle.load(open(scaler_path, 'rb'))
                    break
                except:
                    continue
        
        for model_path in [p for p in model_paths if 'model' in p and 'scaler' not in p]:
            if os.path.exists(model_path):
                try:
                    model = pickle.load(open(model_path, 'rb'))
                    break
                except:
                    continue
        
        if scaler is None or model is None:
            st.warning("⚠️ Pre-trained models not found. Using fallback recommendation system.")
            return None, None
            
        return scaler, model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Fallback recommendation system when models are not available
class FallbackRecommendationSystem:
    def __init__(self):
        self.career_paths = {
            'Medical Doctor': {
                'subjects': ['biology', 'chemistry', 'physics'],
                'weights': {'biology': 0.4, 'chemistry': 0.3, 'physics': 0.3},
                'min_avg': 70,
                'description': 'MBBS, BDS, Veterinary Science, Pharmacy'
            },
            'Software Engineer': {
                'subjects': ['maths', 'bio_cs', 'physics'],
                'weights': {'maths': 0.4, 'bio_cs': 0.4, 'physics': 0.2},
                'min_avg': 65,
                'description': 'Programming, Software Development, Web Development'
            },
            'Data Scientist': {
                'subjects': ['maths', 'bio_cs', 'english'],
                'weights': {'maths': 0.5, 'bio_cs': 0.3, 'english': 0.2},
                'min_avg': 70,
                'description': 'Data Analysis, Machine Learning, AI, Statistics'
            },
            'Engineering': {
                'subjects': ['maths', 'physics', 'chemistry'],
                'weights': {'maths': 0.4, 'physics': 0.3, 'chemistry': 0.3},
                'min_avg': 65,
                'description': 'Civil, Mechanical, Electrical, Electronics Engineering'
            },
            'Teacher': {
                'subjects': ['english', 'tamil'],
                'weights': {'english': 0.5, 'tamil': 0.5},
                'min_avg': 60,
                'description': 'Education, Academic Instruction, Curriculum Development'
            },
            'Business Analyst': {
                'subjects': ['maths', 'english'],
                'weights': {'maths': 0.6, 'english': 0.4},
                'min_avg': 65,
                'description': 'Business Analysis, Consulting, Management'
            },
            'Scientist': {
                'subjects': ['maths', 'physics', 'chemistry', 'biology'],
                'weights': {'maths': 0.25, 'physics': 0.25, 'chemistry': 0.25, 'biology': 0.25},
                'min_avg': 70,
                'description': 'Research, Laboratory Work, Scientific Analysis'
            },
            'Government Officer': {
                'subjects': ['english', 'tamil'],
                'weights': {'english': 0.6, 'tamil': 0.4},
                'min_avg': 55,
                'description': 'Civil Services, Public Administration, Policy Making'
            }
        }
    
    def get_recommendations(self, scores, extracurricular, study_hours, part_time_job):
        recommendations = []
        
        subject_mapping = {
            'tamil': scores['tamil'],
            'english': scores['english'],
            'maths': scores['maths'],
            'bio_cs': scores['bio_cs'],
            'physics': scores['physics'],
            'chemistry': scores['chemistry'],
            'biology': scores['bio_cs']  # Assume bio_cs represents biology for medical careers
        }
        
        for career, details in self.career_paths.items():
            score = 0
            total_weight = 0
            
            for subject, weight in details['weights'].items():
                if subject in subject_mapping:
                    score += subject_mapping[subject] * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_score = score / total_weight
                
                # Apply bonuses
                if extracurricular:
                    weighted_score += 2
                if study_hours > 15:
                    weighted_score += 3
                elif study_hours > 10:
                    weighted_score += 1.5
                
                # Apply penalty for part-time job
                if part_time_job:
                    weighted_score -= 1
                
                # Only include if meets minimum requirements
                if weighted_score >= details['min_avg']:
                    probability = min(weighted_score / 100, 0.95)
                    recommendations.append((career, probability, details['description']))
        
        # Sort by probability
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:5]

# Updated class names without 'Unknown'
class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Software Engineer', 
               'Teacher', 'Business Owner', 'Scientist', 'Banker', 'Writer', 
               'Accountant', 'Designer', 'Construction Engineer', 'Game Developer', 
               'Stock Investor', 'Real Estate Developer', 'Data Scientist', 'Pharmacist']

# Career descriptions
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

# Enhanced recommendation function
def get_recommendations(gender, part_time_job, extracurricular_activities,
                       weekly_self_study_hours, scores, total_score, average_score):
    
    scaler, model = load_models()
    
    if scaler is not None and model is not None:
        # Use the original ML model
        try:
            # Encode categorical variables
            gender_encoded = 1 if gender.lower() == 'female' else 0
            part_time_job_encoded = 1 if part_time_job else 0
            extracurricular_activities_encoded = 1 if extracurricular_activities else 0

            # Create feature array matching the model's expected input
            feature_array = np.array([[gender_encoded, part_time_job_encoded, extracurricular_activities_encoded,
                                     weekly_self_study_hours, scores['maths'], scores['tamil'], scores['physics'],
                                     scores['chemistry'], scores['bio_cs'], scores['english'], 
                                     (scores['tamil'] + scores['english']) / 2,  # geography approximation
                                     total_score, average_score]])

            # Scale features and predict
            scaled_features = scaler.transform(feature_array)
            probabilities = model.predict_proba(scaled_features)

            # Get top recommendations
            top_classes_idx = np.argsort(-probabilities[0])[:5]
            recommendations = []
            
            for idx in top_classes_idx:
                prob = probabilities[0][idx]
                if prob > 0.05:  # Only show >5% probability
                    career_name = class_names[idx] if idx < len(class_names) else "Career Option"
                    description = career_descriptions.get(career_name, "Explore this career path")
                    recommendations.append((career_name, prob, description))
            
            return recommendations, "AI Model"
            
        except Exception as e:
            st.warning(f"ML model prediction failed: {str(e)}. Using fallback system.")
    
    # Use fallback system
    fallback_system = FallbackRecommendationSystem()
    recommendations = fallback_system.get_recommendations(scores, extracurricular_activities, 
                                                        weekly_self_study_hours, part_time_job)
    return recommendations, "Rule-based System"

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
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎓 Career Guidance Recommendation System</h1>', unsafe_allow_html=True)
    
    st.write("""
    ### Welcome to Your Career Path Explorer! 
    This intelligent system analyzes your academic performance across 6 key subjects (Total: 600 marks) 
    and provides personalized career recommendations tailored to your strengths and interests.
    """)
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📋 Personal Information")
        
        # Personal details
        gender = st.selectbox("👤 Gender", ["Male", "Female"], index=0)
        part_time_job = st.selectbox("💼 Part-Time Job", ["Yes", "No"], index=1)
        extracurricular_activities = st.selectbox("🎭 Extracurricular Activities", ["Yes", "No"], index=0)
        weekly_self_study_hours = st.number_input("⏱️ Weekly Self-Study Hours", min_value=0, max_value=100, step=1, value=10)
        
        st.header("📊 Subject Scores (Out of 100 each - Total 600 marks)")
        
        # Subject scores in two columns
        col_sub1, col_sub2 = st.columns(2)
        
        with col_sub1:
            tamil_score = st.number_input("🔤 Tamil", min_value=0, max_value=100, value=75, step=1)
            english_score = st.number_input("📚 English", min_value=0, max_value=100, value=75, step=1)
            math_score = st.number_input("🔢 Mathematics", min_value=0, max_value=100, value=75, step=1)
        
        with col_sub2:
            bio_cs_score = st.number_input("🧬/💻 Biology/Computer Science", min_value=0, max_value=100, value=75, step=1)
            physics_score = st.number_input("⚛️ Physics", min_value=0, max_value=100, value=75, step=1)
            chemistry_score = st.number_input("⚗️ Chemistry", min_value=0, max_value=100, value=75, step=1)
    
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
        
        # Grade calculation
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
                    color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
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
    if st.button("🔍 Get Career Recommendations", type="primary"):
        if average_score < 35:
            st.error("⚠️ Your overall performance is below passing grade. Focus on improving your scores to at least 35% average.")
        else:
            with st.spinner("🤖 Analyzing your profile..."):
                scores_dict = {
                    'tamil': tamil_score,
                    'english': english_score,
                    'maths': math_score,
                    'bio_cs': bio_cs_score,
                    'physics': physics_score,
                    'chemistry': chemistry_score
                }
                
                recommendations, system_type = get_recommendations(
                    gender, part_time_job == "Yes", extracurricular_activities == "Yes",
                    weekly_self_study_hours, scores_dict, total_score, average_score
                )
                
                if recommendations:
                    st.success(f"✅ Analysis Complete using {system_type}!")
                    st.header("🎯 Your Personalized Career Recommendations")
                    
                    for i, rec in enumerate(recommendations, 1):
                        career = rec[0]
                        probability = rec[1]
                        description = rec[2] if len(rec) > 2 else career_descriptions.get(career, "Explore this career path")
                        
                        confidence_percentage = probability * 100
                        
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
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Guidance section
                    st.info("""
                    💡 **Next Steps:**
                    1. Research your top 3 recommended careers in detail
                    2. Connect with professionals in these fields
                    3. Look for internships or job shadowing opportunities
                    4. Focus on improving scores in relevant subjects
                    """)
                else:
                    st.error("Unable to generate recommendations. Please try again.")

if __name__ == '__main__':
    main()
