from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

COMPLETEJOURNEY_URLS = {
    "transactions_sample": "https://raw.githubusercontent.com/bradleyboehmke/completejourney/master/data/transactions_sample.rda",
    "products": "https://raw.githubusercontent.com/bradleyboehmke/completejourney/master/data/products.rda",
    "coupon_redemptions": "https://raw.githubusercontent.com/bradleyboehmke/completejourney/master/data/coupon_redemptions.rda",
    "campaigns": "https://raw.githubusercontent.com/bradleyboehmke/completejourney/master/data/campaigns.rda",
}

CAMPAIGN_CLASSES = [
    "new_customer_onboarding",
    "win_back_reminder",
    "price_led_coupon",
    "cross_sell_bundle",
    "loyalty_reward",
    "seasonal_spotlight",
]

CAMPAIGN_LABELS = {
    "new_customer_onboarding": "New customer onboarding",
    "win_back_reminder": "Win-back reminder",
    "price_led_coupon": "Price-led coupon",
    "cross_sell_bundle": "Cross-sell bundle",
    "loyalty_reward": "Loyalty reward",
    "seasonal_spotlight": "Seasonal spotlight",
}

CAMPAIGN_DESCRIPTIONS = {
    "new_customer_onboarding": "Educate newer shoppers, reinforce value, and encourage a second or third trip.",
    "win_back_reminder": "Reactivate households whose recent shopping activity has slowed down.",
    "price_led_coupon": "Use a discount-led offer for audiences that show strong coupon or promotion sensitivity.",
    "cross_sell_bundle": "Promote a complementary category or bundle to increase basket size.",
    "loyalty_reward": "Reward consistent high-value shoppers and protect their engagement.",
    "seasonal_spotlight": "Align the audience with seasonal demand patterns and timely occasions.",
}

CATEGORY_ALIASES = {
    "beverages": ["soda", "soft drink", "beverage", "beverages", "cola", "coke", "pepsi"],
    "soda": ["soda", "soft drink", "beverage", "cola", "coke", "pepsi"],
    "snacks": ["snack", "chips", "crackers", "popcorn"],
    "dairy": ["dairy", "milk", "cheese", "yogurt"],
    "frozen": ["frozen", "ice cream"],
    "bakery": ["bakery", "bread", "cake"],
    "meat": ["meat", "chicken", "beef", "pork"],
    "produce": ["produce", "fruit", "vegetable"],
    "personal care": ["personal care", "beauty", "shampoo"],
}

OCCASION_CATEGORY_BUNDLES = {
    "game day": ["beverages", "snacks", "meat", "frozen"],
    "game": ["beverages", "snacks", "meat", "frozen"],
    "party": ["beverages", "snacks", "bakery", "frozen"],
    "weekend": ["beverages", "snacks", "meat", "bakery"],
    "back to school": ["snacks", "breakfast", "personal care"],
    "holiday": ["beverages", "snacks", "bakery", "meat"],
}
