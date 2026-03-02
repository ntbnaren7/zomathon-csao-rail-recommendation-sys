"""
CSAO Menu Catalog — 50+ restaurants, 500+ items across 15 cuisines, 5 cities.
Each item has Meal DNA role, pairing info, and rich attributes.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Lucknow"]

CUISINES = [
    "North Indian", "South Indian", "Biryani", "Chinese", "Fast Food",
    "Pizza", "Street Food", "Mughlai", "Desserts", "Healthy",
    "Cafe", "Rolls", "Thali", "Regional", "Cloud Kitchen"
]

MEAL_ROLES = ["protein", "carb", "side", "dessert", "drink", "accompaniment"]

CATEGORIES = [
    "starter", "main_course", "side_dish", "bread", "rice",
    "dessert", "beverage", "accompaniment", "combo", "snack"
]

PRICE_TIERS = {"budget": (30, 150), "mid": (151, 350), "premium": (351, 800)}


# ─────────────────────────────────────────────
# Restaurant Definitions
# ─────────────────────────────────────────────
RESTAURANTS = [
    # North Indian
    {"id": "R001", "name": "Punjab Grill", "city": "Delhi", "cuisine": "North Indian", "price_tier": "premium", "is_chain": False, "rating": 4.3},
    {"id": "R002", "name": "Moti Mahal Delux", "city": "Delhi", "cuisine": "North Indian", "price_tier": "mid", "is_chain": True, "rating": 4.1},
    {"id": "R003", "name": "Bikanervala", "city": "Delhi", "cuisine": "North Indian", "price_tier": "budget", "is_chain": True, "rating": 4.0},
    {"id": "R004", "name": "Dhaba Express", "city": "Mumbai", "cuisine": "North Indian", "price_tier": "budget", "is_chain": False, "rating": 3.8},
    # South Indian
    {"id": "R005", "name": "Saravana Bhavan", "city": "Bangalore", "cuisine": "South Indian", "price_tier": "mid", "is_chain": True, "rating": 4.4},
    {"id": "R006", "name": "Dosa Plaza", "city": "Mumbai", "cuisine": "South Indian", "price_tier": "budget", "is_chain": True, "rating": 3.9},
    {"id": "R007", "name": "A2B Adyar Ananda Bhavan", "city": "Bangalore", "cuisine": "South Indian", "price_tier": "mid", "is_chain": True, "rating": 4.2},
    # Biryani
    {"id": "R008", "name": "Paradise Biryani", "city": "Hyderabad", "cuisine": "Biryani", "price_tier": "mid", "is_chain": True, "rating": 4.5},
    {"id": "R009", "name": "Behrouz Biryani", "city": "Mumbai", "cuisine": "Biryani", "price_tier": "premium", "is_chain": True, "rating": 4.3},
    {"id": "R010", "name": "Shah Ghouse", "city": "Hyderabad", "cuisine": "Biryani", "price_tier": "mid", "is_chain": False, "rating": 4.4},
    {"id": "R011", "name": "Lucknow Biryani House", "city": "Lucknow", "cuisine": "Biryani", "price_tier": "budget", "is_chain": False, "rating": 4.0},
    # Chinese / Indo-Chinese
    {"id": "R012", "name": "Mainland China", "city": "Mumbai", "cuisine": "Chinese", "price_tier": "premium", "is_chain": True, "rating": 4.2},
    {"id": "R013", "name": "Wow Momo", "city": "Bangalore", "cuisine": "Chinese", "price_tier": "budget", "is_chain": True, "rating": 3.9},
    {"id": "R014", "name": "Chowman", "city": "Delhi", "cuisine": "Chinese", "price_tier": "mid", "is_chain": True, "rating": 4.1},
    # Fast Food
    {"id": "R015", "name": "McDonald's", "city": "Mumbai", "cuisine": "Fast Food", "price_tier": "budget", "is_chain": True, "rating": 4.0},
    {"id": "R016", "name": "Burger King", "city": "Delhi", "cuisine": "Fast Food", "price_tier": "budget", "is_chain": True, "rating": 3.8},
    {"id": "R017", "name": "KFC", "city": "Bangalore", "cuisine": "Fast Food", "price_tier": "mid", "is_chain": True, "rating": 4.0},
    # Pizza
    {"id": "R018", "name": "Domino's Pizza", "city": "Mumbai", "cuisine": "Pizza", "price_tier": "mid", "is_chain": True, "rating": 3.9},
    {"id": "R019", "name": "Pizza Hut", "city": "Delhi", "cuisine": "Pizza", "price_tier": "mid", "is_chain": True, "rating": 3.8},
    {"id": "R020", "name": "La Pino'z", "city": "Bangalore", "cuisine": "Pizza", "price_tier": "budget", "is_chain": True, "rating": 4.0},
    # Street Food / Chaat
    {"id": "R021", "name": "Haldiram's", "city": "Delhi", "cuisine": "Street Food", "price_tier": "budget", "is_chain": True, "rating": 4.1},
    {"id": "R022", "name": "Chaat Corner", "city": "Lucknow", "cuisine": "Street Food", "price_tier": "budget", "is_chain": False, "rating": 3.7},
    {"id": "R023", "name": "Bombay Chowpatty", "city": "Mumbai", "cuisine": "Street Food", "price_tier": "budget", "is_chain": False, "rating": 3.9},
    # Mughlai / Kebab
    {"id": "R024", "name": "Karim's", "city": "Delhi", "cuisine": "Mughlai", "price_tier": "mid", "is_chain": False, "rating": 4.5},
    {"id": "R025", "name": "Tunday Kababi", "city": "Lucknow", "cuisine": "Mughlai", "price_tier": "mid", "is_chain": False, "rating": 4.6},
    {"id": "R026", "name": "ITC Bukhara", "city": "Delhi", "cuisine": "Mughlai", "price_tier": "premium", "is_chain": False, "rating": 4.7},
    # Desserts / Bakery
    {"id": "R027", "name": "Theobroma", "city": "Mumbai", "cuisine": "Desserts", "price_tier": "premium", "is_chain": True, "rating": 4.5},
    {"id": "R028", "name": "Baskin Robbins", "city": "Bangalore", "cuisine": "Desserts", "price_tier": "mid", "is_chain": True, "rating": 4.1},
    {"id": "R029", "name": "Karachi Bakery", "city": "Hyderabad", "cuisine": "Desserts", "price_tier": "budget", "is_chain": True, "rating": 4.3},
    # Healthy / Salads
    {"id": "R030", "name": "FreshMenu", "city": "Bangalore", "cuisine": "Healthy", "price_tier": "mid", "is_chain": True, "rating": 4.0},
    {"id": "R031", "name": "Salad Days", "city": "Delhi", "cuisine": "Healthy", "price_tier": "mid", "is_chain": False, "rating": 4.2},
    # Cafe
    {"id": "R032", "name": "Starbucks", "city": "Mumbai", "cuisine": "Cafe", "price_tier": "premium", "is_chain": True, "rating": 4.2},
    {"id": "R033", "name": "Cafe Coffee Day", "city": "Bangalore", "cuisine": "Cafe", "price_tier": "mid", "is_chain": True, "rating": 3.8},
    {"id": "R034", "name": "Chaayos", "city": "Delhi", "cuisine": "Cafe", "price_tier": "budget", "is_chain": True, "rating": 4.0},
    # Rolls / Wraps
    {"id": "R035", "name": "Faasos", "city": "Mumbai", "cuisine": "Rolls", "price_tier": "budget", "is_chain": True, "rating": 3.9},
    {"id": "R036", "name": "Kathi Junction", "city": "Delhi", "cuisine": "Rolls", "price_tier": "budget", "is_chain": True, "rating": 3.7},
    # Thali
    {"id": "R037", "name": "Rajdhani Thali", "city": "Mumbai", "cuisine": "Thali", "price_tier": "mid", "is_chain": True, "rating": 4.3},
    {"id": "R038", "name": "Bhojanam", "city": "Hyderabad", "cuisine": "Thali", "price_tier": "budget", "is_chain": False, "rating": 4.0},
    # Regional
    {"id": "R039", "name": "Oh! Calcutta", "city": "Mumbai", "cuisine": "Regional", "price_tier": "premium", "is_chain": False, "rating": 4.4},
    {"id": "R040", "name": "Rajasthani Rasoi", "city": "Delhi", "cuisine": "Regional", "price_tier": "mid", "is_chain": False, "rating": 4.0},
    # Cloud Kitchen / Fusion
    {"id": "R041", "name": "Box8", "city": "Mumbai", "cuisine": "Cloud Kitchen", "price_tier": "budget", "is_chain": True, "rating": 3.8},
    {"id": "R042", "name": "EatFit", "city": "Bangalore", "cuisine": "Cloud Kitchen", "price_tier": "mid", "is_chain": True, "rating": 4.0},
    {"id": "R043", "name": "Rebel Foods Hub", "city": "Delhi", "cuisine": "Cloud Kitchen", "price_tier": "mid", "is_chain": True, "rating": 3.9},
    # Additional per-city coverage
    {"id": "R044", "name": "Hyderabadi House", "city": "Hyderabad", "cuisine": "North Indian", "price_tier": "mid", "is_chain": False, "rating": 4.1},
    {"id": "R045", "name": "MTR", "city": "Bangalore", "cuisine": "South Indian", "price_tier": "mid", "is_chain": True, "rating": 4.5},
    {"id": "R046", "name": "Aminia", "city": "Lucknow", "cuisine": "Mughlai", "price_tier": "mid", "is_chain": False, "rating": 4.2},
    {"id": "R047", "name": "SubWay", "city": "Mumbai", "cuisine": "Fast Food", "price_tier": "mid", "is_chain": True, "rating": 3.7},
    {"id": "R048", "name": "Natural Ice Cream", "city": "Mumbai", "cuisine": "Desserts", "price_tier": "mid", "is_chain": True, "rating": 4.4},
    {"id": "R049", "name": "Chai Sutta Bar", "city": "Lucknow", "cuisine": "Cafe", "price_tier": "budget", "is_chain": True, "rating": 3.6},
    {"id": "R050", "name": "Biryani Blues", "city": "Delhi", "cuisine": "Biryani", "price_tier": "mid", "is_chain": True, "rating": 4.0},
    {"id": "R051", "name": "Taco Bell", "city": "Mumbai", "cuisine": "Fast Food", "price_tier": "mid", "is_chain": True, "rating": 3.8},
    {"id": "R052", "name": "Sagar Ratna", "city": "Delhi", "cuisine": "South Indian", "price_tier": "mid", "is_chain": True, "rating": 4.0},
]


# ─────────────────────────────────────────────
# Menu Item Templates by Cuisine
# Each template is expanded for each restaurant of that cuisine
# ─────────────────────────────────────────────

def _build_item(item_id, name, restaurant_id, cuisine, category,
                is_veg, base_price, meal_role, spice_level=2,
                is_shareable=False, pairs_with=None, popularity=0.5):
    return {
        "item_id": item_id,
        "name": name,
        "restaurant_id": restaurant_id,
        "cuisine": cuisine,
        "category": category,
        "is_veg": is_veg,
        "price": base_price,
        "meal_role": meal_role,
        "spice_level": spice_level,  # 1-5
        "is_shareable": is_shareable,
        "pairs_with": pairs_with or [],
        "popularity": popularity,  # 0-1
        "avg_rating": round(3.5 + np.random.random() * 1.3, 1),
    }


# Cuisine-specific menu templates (item name, category, is_veg, price_range, meal_role, spice, shareable, popularity)
MENU_TEMPLATES = {
    "North Indian": [
        ("Butter Chicken", "main_course", False, (280, 380), "protein", 2, False, 0.95),
        ("Paneer Butter Masala", "main_course", True, (250, 340), "protein", 2, False, 0.92),
        ("Dal Makhani", "main_course", True, (200, 280), "side", 1, False, 0.88),
        ("Kadai Paneer", "main_course", True, (240, 320), "protein", 3, False, 0.75),
        ("Chicken Tikka", "starter", False, (220, 300), "protein", 3, False, 0.80),
        ("Paneer Tikka", "starter", True, (200, 280), "protein", 3, False, 0.78),
        ("Malai Kofta", "main_course", True, (230, 310), "protein", 1, False, 0.65),
        ("Chole Bhature", "main_course", True, (150, 220), "carb", 2, False, 0.82),
        ("Butter Naan", "bread", True, (40, 70), "carb", 1, False, 0.96),
        ("Garlic Naan", "bread", True, (50, 80), "carb", 1, False, 0.90),
        ("Tandoori Roti", "bread", True, (30, 50), "carb", 1, False, 0.85),
        ("Laccha Paratha", "bread", True, (50, 80), "carb", 1, False, 0.70),
        ("Jeera Rice", "rice", True, (150, 220), "carb", 1, False, 0.78),
        ("Steamed Rice", "rice", True, (100, 160), "carb", 1, False, 0.72),
        ("Raita", "side_dish", True, (60, 100), "accompaniment", 1, False, 0.88),
        ("Green Salad", "side_dish", True, (80, 120), "side", 1, False, 0.55),
        ("Onion Salad", "side_dish", True, (30, 50), "accompaniment", 1, False, 0.60),
        ("Gulab Jamun", "dessert", True, (80, 140), "dessert", 1, True, 0.82),
        ("Rasmalai", "dessert", True, (100, 160), "dessert", 1, False, 0.68),
        ("Kheer", "dessert", True, (90, 130), "dessert", 1, False, 0.55),
        ("Lassi (Sweet)", "beverage", True, (80, 130), "drink", 1, False, 0.75),
        ("Masala Chaas", "beverage", True, (60, 90), "drink", 1, False, 0.65),
        ("Coke / Pepsi", "beverage", True, (40, 60), "drink", 1, False, 0.70),
        ("Papad", "accompaniment", True, (30, 50), "accompaniment", 1, False, 0.72),
        ("Pickle", "accompaniment", True, (20, 40), "accompaniment", 3, False, 0.50),
    ],
    "South Indian": [
        ("Masala Dosa", "main_course", True, (120, 200), "carb", 2, False, 0.95),
        ("Plain Dosa", "main_course", True, (90, 150), "carb", 1, False, 0.80),
        ("Rava Dosa", "main_course", True, (130, 210), "carb", 2, False, 0.72),
        ("Mysore Masala Dosa", "main_course", True, (140, 220), "carb", 3, False, 0.78),
        ("Idli (2 pcs)", "main_course", True, (60, 100), "carb", 1, False, 0.90),
        ("Medu Vada (2 pcs)", "snack", True, (70, 110), "protein", 1, False, 0.82),
        ("Uttapam", "main_course", True, (110, 170), "carb", 2, False, 0.65),
        ("Pongal", "main_course", True, (100, 160), "carb", 1, False, 0.60),
        ("Sambar", "side_dish", True, (50, 90), "side", 2, False, 0.92),
        ("Coconut Chutney", "accompaniment", True, (30, 50), "accompaniment", 1, False, 0.90),
        ("Tomato Chutney", "accompaniment", True, (30, 50), "accompaniment", 2, False, 0.65),
        ("Rasam", "side_dish", True, (50, 80), "side", 2, False, 0.70),
        ("Curd Rice", "rice", True, (90, 140), "carb", 1, False, 0.68),
        ("Lemon Rice", "rice", True, (100, 150), "carb", 1, False, 0.62),
        ("Filter Coffee", "beverage", True, (50, 90), "drink", 1, False, 0.88),
        ("Badam Milk", "beverage", True, (70, 110), "drink", 1, False, 0.55),
        ("Kesari / Halwa", "dessert", True, (60, 100), "dessert", 1, False, 0.60),
        ("Payasam", "dessert", True, (70, 110), "dessert", 1, False, 0.50),
        ("Upma", "main_course", True, (80, 130), "carb", 1, False, 0.55),
    ],
    "Biryani": [
        ("Chicken Biryani", "rice", False, (250, 400), "protein", 3, True, 0.97),
        ("Mutton Biryani", "rice", False, (350, 500), "protein", 3, True, 0.88),
        ("Veg Biryani", "rice", True, (200, 300), "carb", 2, True, 0.75),
        ("Egg Biryani", "rice", False, (200, 320), "protein", 3, True, 0.70),
        ("Paneer Biryani", "rice", True, (230, 340), "protein", 2, True, 0.65),
        ("Hyderabadi Dum Biryani", "rice", False, (300, 450), "protein", 4, True, 0.92),
        ("Mirchi Ka Salan", "side_dish", True, (80, 140), "side", 4, False, 0.90),
        ("Bagara Baingan", "side_dish", True, (100, 160), "side", 3, False, 0.55),
        ("Raita", "side_dish", True, (50, 90), "accompaniment", 1, False, 0.92),
        ("Shorba", "starter", False, (70, 120), "side", 2, False, 0.60),
        ("Chicken 65", "starter", False, (180, 260), "protein", 4, False, 0.78),
        ("Kebab Platter", "starter", False, (250, 380), "protein", 3, True, 0.72),
        ("Gulab Jamun (2pc)", "dessert", True, (60, 100), "dessert", 1, False, 0.82),
        ("Double Ka Meetha", "dessert", True, (80, 130), "dessert", 1, False, 0.55),
        ("Phirni", "dessert", True, (70, 110), "dessert", 1, False, 0.50),
        ("Thumbs Up / Coke", "beverage", True, (40, 60), "drink", 1, False, 0.75),
        ("Lassi", "beverage", True, (70, 110), "drink", 1, False, 0.65),
        ("Rumali Roti", "bread", True, (40, 60), "carb", 1, False, 0.45),
        ("Papad", "accompaniment", True, (20, 40), "accompaniment", 1, False, 0.60),
    ],
    "Chinese": [
        ("Veg Manchurian", "starter", True, (180, 260), "protein", 3, False, 0.85),
        ("Chicken Manchurian", "starter", False, (220, 300), "protein", 3, False, 0.82),
        ("Veg Fried Rice", "rice", True, (160, 230), "carb", 2, False, 0.90),
        ("Chicken Fried Rice", "rice", False, (200, 280), "carb", 2, False, 0.88),
        ("Egg Fried Rice", "rice", False, (170, 240), "carb", 2, False, 0.72),
        ("Hakka Noodles", "main_course", True, (160, 230), "carb", 2, False, 0.88),
        ("Chilli Chicken", "main_course", False, (230, 310), "protein", 4, False, 0.85),
        ("Chilli Paneer", "main_course", True, (210, 290), "protein", 4, False, 0.80),
        ("Spring Roll (4pc)", "starter", True, (120, 180), "side", 2, True, 0.78),
        ("Momos (Steamed/Fried)", "starter", True, (100, 160), "side", 2, True, 0.92),
        ("Chicken Momos", "starter", False, (130, 190), "side", 2, True, 0.90),
        ("Sweet Corn Soup", "starter", True, (100, 150), "side", 1, False, 0.72),
        ("Hot & Sour Soup", "starter", False, (110, 160), "side", 3, False, 0.68),
        ("Schezwan Fried Rice", "rice", True, (180, 250), "carb", 4, False, 0.75),
        ("Honey Chilli Potato", "starter", True, (150, 220), "side", 3, False, 0.78),
        ("Coke / Thumbs Up", "beverage", True, (40, 60), "drink", 1, False, 0.68),
        ("Iced Tea", "beverage", True, (70, 110), "drink", 1, False, 0.55),
        ("Brownie", "dessert", True, (80, 130), "dessert", 1, False, 0.45),
    ],
    "Fast Food": [
        ("McChicken / Chicken Burger", "main_course", False, (120, 200), "protein", 2, False, 0.90),
        ("Veg Burger", "main_course", True, (80, 150), "protein", 1, False, 0.82),
        ("Chicken Nuggets (6pc)", "starter", False, (130, 200), "protein", 1, True, 0.78),
        ("French Fries (Regular)", "side_dish", True, (80, 130), "side", 1, True, 0.92),
        ("French Fries (Large)", "side_dish", True, (120, 170), "side", 1, True, 0.70),
        ("Chicken Wings (4pc)", "starter", False, (150, 230), "protein", 3, True, 0.75),
        ("Veg Wrap", "main_course", True, (100, 160), "carb", 1, False, 0.65),
        ("Chicken Wrap", "main_course", False, (130, 200), "protein", 2, False, 0.68),
        ("Hash Brown", "side_dish", True, (50, 80), "side", 1, False, 0.60),
        ("Coke (Medium)", "beverage", True, (60, 90), "drink", 1, False, 0.85),
        ("Coke (Large)", "beverage", True, (90, 120), "drink", 1, False, 0.55),
        ("Fanta / Sprite", "beverage", True, (60, 90), "drink", 1, False, 0.50),
        ("McFlurry / Sundae", "dessert", True, (80, 140), "dessert", 1, False, 0.72),
        ("Apple Pie", "dessert", True, (50, 80), "dessert", 1, False, 0.45),
        ("Piri Piri Fries", "side_dish", True, (100, 150), "side", 3, True, 0.55),
    ],
    "Pizza": [
        ("Margherita Pizza (Medium)", "main_course", True, (200, 350), "carb", 1, True, 0.88),
        ("Farmhouse Pizza (Medium)", "main_course", True, (300, 450), "carb", 1, True, 0.80),
        ("Pepperoni Pizza (Medium)", "main_course", False, (350, 500), "protein", 2, True, 0.82),
        ("Chicken Tikka Pizza", "main_course", False, (350, 480), "protein", 2, True, 0.75),
        ("Paneer Makhani Pizza", "main_course", True, (300, 430), "protein", 1, True, 0.70),
        ("Garlic Breadsticks (4pc)", "side_dish", True, (120, 180), "side", 1, True, 0.90),
        ("Stuffed Garlic Bread", "side_dish", True, (160, 230), "side", 1, True, 0.78),
        ("Pasta Alfredo", "main_course", True, (200, 300), "carb", 1, False, 0.65),
        ("Chicken Penne Pasta", "main_course", False, (250, 350), "protein", 1, False, 0.60),
        ("Lava Cake", "dessert", True, (100, 160), "dessert", 1, False, 0.82),
        ("Brownie Fantasy", "dessert", True, (120, 180), "dessert", 1, False, 0.68),
        ("Coke (500ml)", "beverage", True, (60, 90), "drink", 1, False, 0.80),
        ("Iced Tea", "beverage", True, (80, 120), "drink", 1, False, 0.50),
        ("Cheesy Dip", "accompaniment", True, (30, 50), "accompaniment", 1, False, 0.55),
    ],
    "Street Food": [
        ("Pani Puri (6pc)", "snack", True, (60, 100), "side", 4, True, 0.92),
        ("Bhel Puri", "snack", True, (70, 110), "side", 3, False, 0.85),
        ("Sev Puri", "snack", True, (70, 110), "side", 3, False, 0.72),
        ("Dahi Puri", "snack", True, (80, 120), "side", 2, False, 0.68),
        ("Samosa (2pc)", "snack", True, (40, 70), "side", 2, True, 0.90),
        ("Kachori (2pc)", "snack", True, (50, 80), "side", 3, True, 0.65),
        ("Vada Pav", "main_course", True, (30, 60), "carb", 2, False, 0.88),
        ("Pav Bhaji", "main_course", True, (100, 160), "carb", 2, False, 0.85),
        ("Chole Tikki", "main_course", True, (80, 130), "protein", 2, False, 0.72),
        ("Aloo Tikki", "snack", True, (50, 80), "side", 2, False, 0.68),
        ("Jalebi (250g)", "dessert", True, (70, 120), "dessert", 1, True, 0.78),
        ("Rabri", "dessert", True, (60, 100), "dessert", 1, False, 0.50),
        ("Masala Chai", "beverage", True, (20, 40), "drink", 1, False, 0.80),
        ("Lassi", "beverage", True, (50, 80), "drink", 1, False, 0.72),
        ("Nimbu Pani", "beverage", True, (30, 50), "drink", 1, False, 0.65),
    ],
    "Mughlai": [
        ("Seekh Kebab", "starter", False, (200, 300), "protein", 3, False, 0.90),
        ("Galouti Kebab", "starter", False, (250, 380), "protein", 2, False, 0.85),
        ("Tunday Kebab", "starter", False, (230, 350), "protein", 2, False, 0.88),
        ("Chicken Tikka", "starter", False, (200, 280), "protein", 3, False, 0.82),
        ("Mutton Korma", "main_course", False, (300, 450), "protein", 2, False, 0.80),
        ("Butter Chicken", "main_course", False, (280, 380), "protein", 2, False, 0.90),
        ("Nihari", "main_course", False, (280, 400), "protein", 3, False, 0.72),
        ("Rumali Roti", "bread", True, (30, 50), "carb", 1, False, 0.88),
        ("Sheermal", "bread", True, (40, 70), "carb", 1, False, 0.65),
        ("Naan", "bread", True, (40, 60), "carb", 1, False, 0.92),
        ("Biryani (Chicken)", "rice", False, (300, 450), "protein", 3, True, 0.85),
        ("Shahi Tukda", "dessert", True, (100, 160), "dessert", 1, False, 0.72),
        ("Phirni", "dessert", True, (80, 120), "dessert", 1, False, 0.60),
        ("Rooh Afza", "beverage", True, (50, 80), "drink", 1, False, 0.55),
        ("Lassi", "beverage", True, (60, 90), "drink", 1, False, 0.65),
    ],
    "Desserts": [
        ("Chocolate Truffle Cake (Slice)", "dessert", True, (150, 250), "dessert", 1, False, 0.90),
        ("Red Velvet Cake (Slice)", "dessert", True, (160, 260), "dessert", 1, False, 0.82),
        ("Brownie", "dessert", True, (100, 170), "dessert", 1, False, 0.88),
        ("Cheesecake", "dessert", True, (180, 280), "dessert", 1, False, 0.75),
        ("Tiramisu", "dessert", True, (200, 300), "dessert", 1, False, 0.65),
        ("Gulab Jamun (4pc)", "dessert", True, (100, 160), "dessert", 1, True, 0.78),
        ("Rasgulla (4pc)", "dessert", True, (80, 130), "dessert", 1, True, 0.68),
        ("Ice Cream Scoop", "dessert", True, (80, 140), "dessert", 1, False, 0.85),
        ("Kulfi", "dessert", True, (70, 120), "dessert", 1, False, 0.72),
        ("Cookie (2pc)", "dessert", True, (60, 100), "dessert", 1, False, 0.60),
        ("Cold Coffee", "beverage", True, (120, 180), "drink", 1, False, 0.80),
        ("Hot Chocolate", "beverage", True, (130, 200), "drink", 1, False, 0.55),
    ],
    "Healthy": [
        ("Caesar Salad", "main_course", True, (200, 300), "protein", 1, False, 0.82),
        ("Grilled Chicken Salad", "main_course", False, (250, 350), "protein", 1, False, 0.85),
        ("Quinoa Bowl", "main_course", True, (250, 350), "carb", 1, False, 0.70),
        ("Poke Bowl", "main_course", False, (300, 400), "protein", 1, False, 0.65),
        ("Smoothie Bowl", "main_course", True, (200, 300), "carb", 1, False, 0.60),
        ("Chicken Wrap (Whole Wheat)", "main_course", False, (180, 260), "protein", 1, False, 0.75),
        ("Fruit Bowl", "side_dish", True, (120, 180), "side", 1, False, 0.68),
        ("Protein Shake", "beverage", True, (150, 220), "drink", 1, False, 0.72),
        ("Green Smoothie", "beverage", True, (130, 200), "drink", 1, False, 0.65),
        ("Cold Pressed Juice", "beverage", True, (120, 180), "drink", 1, False, 0.60),
        ("Energy Bar", "snack", True, (80, 130), "side", 1, False, 0.45),
        ("Yogurt Parfait", "dessert", True, (120, 180), "dessert", 1, False, 0.55),
    ],
    "Cafe": [
        ("Cappuccino", "beverage", True, (120, 200), "drink", 1, False, 0.90),
        ("Latte", "beverage", True, (130, 210), "drink", 1, False, 0.85),
        ("Americano", "beverage", True, (100, 170), "drink", 1, False, 0.70),
        ("Cold Coffee", "beverage", True, (140, 220), "drink", 1, False, 0.82),
        ("Chai Latte", "beverage", True, (100, 160), "drink", 1, False, 0.68),
        ("Masala Chai", "beverage", True, (40, 80), "drink", 1, False, 0.88),
        ("Club Sandwich", "main_course", False, (180, 260), "protein", 1, False, 0.80),
        ("Paneer Sandwich", "main_course", True, (150, 220), "protein", 1, False, 0.72),
        ("Croissant", "snack", True, (80, 140), "carb", 1, False, 0.65),
        ("Muffin", "snack", True, (80, 130), "dessert", 1, False, 0.72),
        ("Cookie", "snack", True, (60, 100), "dessert", 1, False, 0.60),
        ("Brownie", "dessert", True, (100, 160), "dessert", 1, False, 0.75),
        ("Cheesecake Slice", "dessert", True, (160, 250), "dessert", 1, False, 0.58),
    ],
    "Rolls": [
        ("Chicken Kathi Roll", "main_course", False, (130, 200), "protein", 2, False, 0.92),
        ("Egg Kathi Roll", "main_course", False, (100, 160), "protein", 2, False, 0.80),
        ("Paneer Kathi Roll", "main_course", True, (120, 180), "protein", 2, False, 0.78),
        ("Double Chicken Roll", "main_course", False, (180, 260), "protein", 2, False, 0.72),
        ("Soya Chaap Roll", "main_course", True, (110, 170), "protein", 2, False, 0.55),
        ("Fries", "side_dish", True, (80, 120), "side", 1, True, 0.82),
        ("Coleslaw", "side_dish", True, (50, 80), "side", 1, False, 0.45),
        ("Coke / Pepsi", "beverage", True, (40, 60), "drink", 1, False, 0.72),
        ("Iced Tea", "beverage", True, (60, 90), "drink", 1, False, 0.50),
        ("Brownie", "dessert", True, (70, 110), "dessert", 1, False, 0.40),
    ],
    "Thali": [
        ("Veg Thali", "combo", True, (200, 350), "carb", 2, False, 0.92),
        ("Non-Veg Thali", "combo", False, (280, 450), "protein", 2, False, 0.85),
        ("Special Thali", "combo", True, (300, 500), "carb", 2, False, 0.70),
        ("Mini Thali", "combo", True, (150, 250), "carb", 2, False, 0.65),
        ("Extra Roti (2pc)", "bread", True, (30, 50), "carb", 1, False, 0.80),
        ("Extra Dal", "side_dish", True, (60, 90), "side", 1, False, 0.55),
        ("Sweet Dish", "dessert", True, (50, 80), "dessert", 1, False, 0.72),
        ("Papad", "accompaniment", True, (20, 30), "accompaniment", 1, False, 0.78),
        ("Lassi", "beverage", True, (60, 90), "drink", 1, False, 0.62),
        ("Chaas", "beverage", True, (40, 60), "drink", 1, False, 0.55),
    ],
    "Regional": [
        ("Fish Curry (Bengali)", "main_course", False, (250, 380), "protein", 2, False, 0.80),
        ("Kosha Mangsho", "main_course", False, (300, 420), "protein", 3, False, 0.78),
        ("Luchi (4pc)", "bread", True, (60, 100), "carb", 1, False, 0.75),
        ("Mishti Doi", "dessert", True, (60, 90), "dessert", 1, False, 0.82),
        ("Sandesh (4pc)", "dessert", True, (80, 130), "dessert", 1, True, 0.72),
        ("Dal Baati Churma", "main_course", True, (180, 280), "carb", 2, False, 0.68),
        ("Laal Maas", "main_course", False, (300, 450), "protein", 5, False, 0.65),
        ("Dhokla", "snack", True, (80, 120), "side", 1, True, 0.70),
        ("Thepla (4pc)", "bread", True, (60, 90), "carb", 1, False, 0.55),
        ("Steamed Rice", "rice", True, (80, 120), "carb", 1, False, 0.72),
        ("Rasgulla", "dessert", True, (70, 110), "dessert", 1, True, 0.68),
        ("Aam Panna", "beverage", True, (50, 80), "drink", 1, False, 0.55),
    ],
    "Cloud Kitchen": [
        ("Chicken Rice Bowl", "combo", False, (180, 260), "protein", 2, False, 0.85),
        ("Paneer Rice Bowl", "combo", True, (160, 230), "protein", 2, False, 0.78),
        ("Egg Fried Rice", "rice", False, (140, 200), "carb", 2, False, 0.72),
        ("Rajma Rice Bowl", "combo", True, (140, 200), "carb", 1, False, 0.68),
        ("Butter Chicken Bowl", "combo", False, (200, 300), "protein", 2, False, 0.80),
        ("Wrap / Roll", "main_course", True, (120, 180), "carb", 1, False, 0.65),
        ("Fries / Wedges", "side_dish", True, (80, 120), "side", 1, True, 0.72),
        ("Coleslaw", "side_dish", True, (50, 70), "side", 1, False, 0.40),
        ("Coke / Pepsi", "beverage", True, (40, 60), "drink", 1, False, 0.70),
        ("Brownie", "dessert", True, (70, 110), "dessert", 1, False, 0.55),
        ("Cookie", "dessert", True, (40, 70), "dessert", 1, False, 0.42),
    ],
}

# Cuisine-specific ideal meal templates (Meal DNA targets)
MEAL_TEMPLATES = {
    "North Indian": {
        "dinner":    {"protein": 1.0, "carb": 1.0, "side": 0.8, "dessert": 0.5, "drink": 0.7, "accompaniment": 0.3},
        "lunch":     {"protein": 1.0, "carb": 1.0, "side": 0.6, "dessert": 0.3, "drink": 0.5, "accompaniment": 0.2},
        "breakfast": {"protein": 0.5, "carb": 1.0, "side": 0.3, "dessert": 0.2, "drink": 0.8, "accompaniment": 0.1},
    },
    "South Indian": {
        "dinner":    {"protein": 0.5, "carb": 1.0, "side": 1.0, "dessert": 0.3, "drink": 0.5, "accompaniment": 0.5},
        "lunch":     {"protein": 0.5, "carb": 1.0, "side": 1.0, "dessert": 0.3, "drink": 0.5, "accompaniment": 0.5},
        "breakfast": {"protein": 0.3, "carb": 1.0, "side": 0.8, "dessert": 0.2, "drink": 0.8, "accompaniment": 0.5},
    },
    "Biryani": {
        "dinner":    {"protein": 1.0, "carb": 0.3, "side": 0.8, "dessert": 0.5, "drink": 0.7, "accompaniment": 0.4},
        "lunch":     {"protein": 1.0, "carb": 0.2, "side": 0.7, "dessert": 0.3, "drink": 0.6, "accompaniment": 0.3},
    },
    "Chinese": {
        "dinner":    {"protein": 1.0, "carb": 1.0, "side": 0.7, "dessert": 0.3, "drink": 0.5, "accompaniment": 0.1},
        "lunch":     {"protein": 0.8, "carb": 1.0, "side": 0.5, "dessert": 0.2, "drink": 0.4, "accompaniment": 0.1},
    },
    "Fast Food": {
        "any":       {"protein": 1.0, "carb": 0.3, "side": 0.8, "dessert": 0.4, "drink": 0.9, "accompaniment": 0.1},
    },
    "Pizza": {
        "any":       {"protein": 0.3, "carb": 1.0, "side": 0.7, "dessert": 0.4, "drink": 0.8, "accompaniment": 0.2},
    },
    "Street Food": {
        "snack":     {"protein": 0.3, "carb": 0.5, "side": 1.0, "dessert": 0.4, "drink": 0.7, "accompaniment": 0.1},
        "any":       {"protein": 0.4, "carb": 0.6, "side": 1.0, "dessert": 0.3, "drink": 0.6, "accompaniment": 0.1},
    },
    "Mughlai": {
        "dinner":    {"protein": 1.0, "carb": 1.0, "side": 0.5, "dessert": 0.6, "drink": 0.5, "accompaniment": 0.2},
        "lunch":     {"protein": 1.0, "carb": 0.8, "side": 0.4, "dessert": 0.4, "drink": 0.4, "accompaniment": 0.2},
    },
    "Desserts": {
        "any":       {"protein": 0.0, "carb": 0.0, "side": 0.0, "dessert": 1.0, "drink": 0.7, "accompaniment": 0.0},
    },
    "Healthy": {
        "any":       {"protein": 1.0, "carb": 0.5, "side": 0.5, "dessert": 0.3, "drink": 0.8, "accompaniment": 0.0},
    },
    "Cafe": {
        "any":       {"protein": 0.5, "carb": 0.5, "side": 0.2, "dessert": 0.5, "drink": 1.0, "accompaniment": 0.0},
    },
    "Rolls": {
        "any":       {"protein": 1.0, "carb": 0.2, "side": 0.5, "dessert": 0.3, "drink": 0.7, "accompaniment": 0.0},
    },
    "Thali": {
        "any":       {"protein": 0.8, "carb": 1.0, "side": 0.8, "dessert": 0.7, "drink": 0.5, "accompaniment": 0.5},
    },
    "Regional": {
        "dinner":    {"protein": 1.0, "carb": 1.0, "side": 0.6, "dessert": 0.6, "drink": 0.4, "accompaniment": 0.2},
        "lunch":     {"protein": 0.8, "carb": 1.0, "side": 0.5, "dessert": 0.4, "drink": 0.3, "accompaniment": 0.2},
    },
    "Cloud Kitchen": {
        "any":       {"protein": 0.8, "carb": 0.5, "side": 0.5, "dessert": 0.3, "drink": 0.6, "accompaniment": 0.0},
    },
}


class MenuCatalog:
    """Generates and manages the full restaurant menu catalog."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.items: List[dict] = []
        self.restaurants_df = pd.DataFrame(RESTAURANTS)
        self._generate_all_items()

    def _price_for_tier(self, base_range: tuple, price_tier: str) -> int:
        """Adjust item price based on restaurant price tier."""
        lo, hi = base_range
        multiplier = {"budget": 0.75, "mid": 1.0, "premium": 1.4}[price_tier]
        adjusted_lo = int(lo * multiplier)
        adjusted_hi = int(hi * multiplier)
        return int(self.rng.integers(adjusted_lo, adjusted_hi + 1))

    def _generate_all_items(self):
        """Generate items for every restaurant from cuisine templates."""
        item_counter = 0
        for rest in RESTAURANTS:
            cuisine = rest["cuisine"]
            templates = MENU_TEMPLATES.get(cuisine, [])
            if not templates:
                continue

            # Each restaurant gets 70-100% of the cuisine template items
            # (simulates menu variation between restaurants of same cuisine)
            n_items = max(5, int(len(templates) * self.rng.uniform(0.7, 1.0)))
            selected = self.rng.choice(len(templates), size=min(n_items, len(templates)), replace=False)

            for idx in selected:
                tmpl = templates[idx]
                name, category, is_veg, price_range, meal_role, spice, shareable, pop = tmpl
                item_counter += 1
                item = _build_item(
                    item_id=f"I{item_counter:04d}",
                    name=name,
                    restaurant_id=rest["id"],
                    cuisine=cuisine,
                    category=category,
                    is_veg=is_veg,
                    base_price=self._price_for_tier(price_range, rest["price_tier"]),
                    meal_role=meal_role,
                    spice_level=spice,
                    is_shareable=shareable,
                    popularity=round(pop + self.rng.uniform(-0.1, 0.1), 2),
                )
                self.items.append(item)

    def get_items_df(self) -> pd.DataFrame:
        """Return all items as a DataFrame."""
        df = pd.DataFrame(self.items)
        df["popularity"] = df["popularity"].clip(0.05, 1.0)
        return df

    def get_restaurant_menu(self, restaurant_id: str) -> pd.DataFrame:
        """Get all items for a specific restaurant."""
        df = self.get_items_df()
        return df[df["restaurant_id"] == restaurant_id].copy()

    def get_restaurants_df(self) -> pd.DataFrame:
        """Return all restaurants as a DataFrame."""
        return self.restaurants_df.copy()

    def get_meal_template(self, cuisine: str, meal_period: str = "dinner") -> dict:
        """Get the ideal Meal DNA template for a cuisine + meal period."""
        templates = MEAL_TEMPLATES.get(cuisine, {})
        if meal_period in templates:
            return templates[meal_period]
        return templates.get("any", {r: 0.5 for r in MEAL_ROLES})

    def save(self, output_dir: str = "data/generated"):
        """Save catalog to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.get_items_df().to_csv(f"{output_dir}/items.csv", index=False)
        self.get_restaurants_df().to_csv(f"{output_dir}/restaurants.csv", index=False)
        print(f"Saved {len(self.items)} items across {len(RESTAURANTS)} restaurants")


if __name__ == "__main__":
    catalog = MenuCatalog(seed=42)
    items_df = catalog.get_items_df()
    rest_df = catalog.get_restaurants_df()

    print(f"Total restaurants: {len(rest_df)}")
    print(f"Total menu items:  {len(items_df)}")
    print(f"\nItems per cuisine:")
    print(items_df.groupby("cuisine").size().sort_values(ascending=False))
    print(f"\nMeal role distribution:")
    print(items_df.groupby("meal_role").size().sort_values(ascending=False))
    print(f"\nVeg/Non-Veg split:")
    print(items_df["is_veg"].value_counts())
    print(f"\nPrice stats:")
    print(items_df["price"].describe())

    catalog.save()
