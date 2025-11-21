# SentimentAnalystSchool

Dự án Phân tích Cảm xúc và Trích xuất Khía cạnh (Aspect-Based Sentiment Analysis) cho các đánh giá sản phẩm.

## 📖 Giới thiệu

Repository này chứa các bài thực hành và đồ án về phân tích cảm xúc dựa trên khía cạnh (ABSA - Aspect-Based Sentiment Analysis). Mục tiêu là phân loại cảm xúc (tích cực, tiêu cực, trung tính, xung đột) cho từng khía cạnh cụ thể trong đánh giá của khách hàng.

## 📁 Cấu trúc Dự án

```
SentimentAnalystSchool/
├── Lab1/                    # Bài thực hành 1: EDA & Traditional ML
│   ├── Lab1.ipynb          # Phân tích dữ liệu và ML cơ bản
│   ├── Lab1_RNN_LSTM.ipynb # Deep Learning với RNN/LSTM
│   └── Dataset/            # Dữ liệu Restaurant và Laptop
│       ├── Restaurant/     # Reviews nhà hàng (SemEval)
│       └── Laptop/         # Reviews laptop (SemEval)
│
├── Lab2/                    # Bài thực hành 2: Transfer Learning
│   ├── Lab2.ipynb          # Fine-tuning BERT cho ABSA
│   └── Dataset/            # Dữ liệu tương tự Lab1
│
|
│
├── MidtermExam/            # Đồ án giữa kỳ
│   ├── aspecttermextraction.ipynb      # Trích xuất khía cạnh
│   ├── AspectTermExtraction_API.ipynb  # API extraction
│   ├── ConvertFile.ipynb               # Chuyển đổi định dạng
│   ├── 10krows.json        # Dữ liệu Electronics reviews
│   ├── reviews.csv         # Dữ liệu đã xử lý
│   ├── DatasetMain/        # Dữ liệu gốc (Amazon, Hotels, Coursera)
│   ├── DatasetUnder80kWord/# Dữ liệu lọc (review < 80 từ)
│   └── Dataset_Laptop_Restaurant/  # ABSA16 SemEval data
│
└── README.md               # File này
```

## 🎯 Các Bài Thực hành

### Lab 1: Exploratory Data Analysis & Traditional ML
**Mục tiêu:** Làm quen với dữ liệu ABSA và các phương pháp ML cơ bản

**Nội dung:**
- Phân tích khám phá dữ liệu (EDA)
  - Phân bố cảm xúc (negative, neutral, positive, conflict)
  - Thống kê độ dài câu
  - Phân tích số lượng khía cạnh trên mỗi câu
- Tiền xử lý văn bản (tokenization, stopwords removal, stemming)
- Feature extraction: TF-IDF
- Traditional ML models:
  - Naive Bayes
  - Logistic Regression
  - SVM (Support Vector Machine)
  - KNN (K-Nearest Neighbors)
- Deep Learning: RNN/LSTM

**Dataset:** SemEval Restaurant & Laptop reviews

**Kết quả chính:** 
- Dữ liệu thiên về cảm xúc tích cực
- Độ dài câu không liên quan đến cảm xúc
- Gần 1/3 số câu có nhiều hơn một khía cạnh

### Lab 2: Transfer Learning với BERT
**Mục tiêu:** Fine-tuning mô hình BERT cho bài toán ABSA

**Nội dung:**
- Tokenization với BERT tokenizer
- Tạo dataset với format phù hợp cho BERT
- Fine-tuning pre-trained BERT model
- Đánh giá mô hình trên test set
- So sánh với traditional ML

**Model:** `bert-base-uncased` từ Hugging Face

**Dataset split:**
- Train: 80%
- Validation: 10%
- Test: 10%

**Labels:**
- 0: negative
- 1: neutral
- 2: positive
- 3: conflict

### Midterm Exam: Aspect Term Extraction & Sentiment Analysis
**Mục tiêu:** Trích xuất aspect terms và phân tích cảm xúc trên dataset thực tế

**Nội dung:**
- Chuyển đổi dữ liệu từ JSON sang CSV
- Trích xuất aspect terms từ reviews
- Xử lý dữ liệu quy mô lớn (10k+ reviews)
- Lọc dữ liệu theo tiêu chí (reviews < 80 từ)

**Datasets:**
- Amazon Electronics Reviews
Dataset service:
- Hotels reviews
- Amazons reviews
Dataset Main:
- SemEval 2016 ABSA data

## 🚀 Cài đặt và Chạy

### Yêu cầu
```bash
Python 3.8+
pip install -r requirements.txt
```