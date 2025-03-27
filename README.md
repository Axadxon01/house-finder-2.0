# house-finder-2.0
# House Finder Pro

**House Finder Pro** is a professional, multi-featured web application built with Streamlit to help users search for houses, list their own properties, predict house prices, and manage real-time announcements. It includes user authentication (manual and Google OAuth), interactive map visualization, interest tracking, multi-language support, and a light/dark mode toggle.

## Features
- **Multi-Page Interface**: Navigate between Home, Search, Sell, Profile, and Announcements pages.
- **User Authentication**: Register/login manually or via Google OAuth.
- **House Search**: Filter houses by budget, bedrooms, year built, and more, with map visualization.
- **Sell Houses**: List properties with image uploads and limited-time expirations.
- **Price Prediction**: Predict house prices using an XGBoost model.
- **Real-Time Updates**: Notifications for expiring listings via polling.
- **Map Visualization**: View house locations on an interactive Folium map.
- **Interest Tracking**: Users can express interest in listings, tracked in the database.
- **Multi-Language Support**: Available in English, O‘zbek, Русский, and Español.
- **Light/Dark Mode**: Toggle between themes for a personalized experience.
- **Error Handling**: Robust logging and user feedback for errors.

## Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)
- A Google Cloud project with OAuth 2.0 credentials (for Google login)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/house-finder-pro.git
cd house-finder-pro
