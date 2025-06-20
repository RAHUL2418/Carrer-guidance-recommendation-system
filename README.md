# Carrer-guidance-recommendation-system
# **CareerPath Recommender 📚✨**  
**A Smart Educational Guidance System for Students**

**CareerPath Recommender**, an interactive recommendation system that helps university students choose the most suitable career paths and educational courses. By analyzing academic performance, personal attributes, and interests, this tool provides personalized career suggestions using machine learning.

---

## **🚀 Features**
- **Personalized Career Recommendations**: Suggests the top three career paths or educational directions based on user profile and scores.
- **Real-Time Calculations**: Automatically computes total and average scores based on subject inputs.
- **Diverse Career Suggestions**: Offers a broad range of career options, from STEM to creative and business fields.
- **User-Friendly Interface**: Built with [Streamlit](https://streamlit.io/) for an engaging and easy-to-use experience.

---

## **🛠️ Tech Stack**
- **Language**: Python
- **Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**:
  - Pre-trained Scikit-learn model for career path prediction.
  - Scaler for input normalization.
- **Other Tools**:
  - **NumPy** for numerical operations.
  - **Pickle** for saving and loading the trained model and scaler.

---

## **📑 How It Works**
1. **User Inputs**: Provide information such as gender, study hours, part-time job status, extracurricular activities, and academic scores (Math, Physics, Chemistry, etc.).
2. **Dynamic Calculations**: The total and average scores are calculated automatically as users enter their scores.
3. **Career Prediction**: A pre-trained machine learning model predicts the top three career paths based on the input data.
4. **Results Displayed**: Career recommendations are shown along with the predicted probabilities.

---

## **🔧 Installation**
Follow the steps below to run the CareerPath Recommender locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Yashpurbhe123/CareerPath_Recommender.git
   
   cd CareerPath_Recommender
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

   The application will launch in your web browser.

---

## **📊 Example Input and Output**
### Example Input:
- **Gender**: Female
- **Weekly Study Hours**: 10
- **Math Score**: 85
- **Physics Score**: 80
- **Chemistry Score**: 75
- **Other Subjects**: Provided by the user

### Example Output:
- **Lawyer** (Probability: 85%)
- **Doctor** (Probability: 70%)
- **Scientist** (Probability: 65%)

---
