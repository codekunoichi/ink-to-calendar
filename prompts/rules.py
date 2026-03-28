SCHEDULING_RULES = {

    # Time blocks: when each category of task is allowed to be scheduled
    "category_windows": {
        "work":     {"days": ["Mon","Tue","Wed","Thu","Fri"], 
                     "start": "09:00", "end": "17:00"},
        "deep_work": {"days": ["Mon","Tue","Wed","Thu","Fri"], 
                      "start": "06:00", "end": "12:00"},
        "chore":    {"days": ["Mon","Tue","Wed","Thu","Fri","Sat"], 
                     "start": "17:30", "end": "21:00"},
        "errand":   {"days": ["Mon","Tue","Wed","Thu","Fri","Sat"], 
                     "start": "12:00", "end": "19:00"},
        "health":   {"days": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], 
                     "start": "06:00", "end": "08:00"},
        "personal": {"days": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], 
                     "start": "18:00", "end": "21:00"},
        "unknown":  {"days": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], 
                     "start": "09:00", "end": "20:00"},
    },

    # Keyword overrides: if the task text contains these words,
    # force-assign to a specific window regardless of category_hint
    "keyword_overrides": {
        "laundry":   "chore",
        "dishes":    "chore",
        "groceries": "errand",
        "sam's":     "errand",
        "costco":    "errand",
        "workout":   "health",
        "gym":       "health",
        "walk":      "health",
        "email":     "work",
        "call":      "work",
        "meeting":   "work",
    },

    # Preferred shopping day: scheduler tries this day first for errands
    "preferred_shopping_day": "Sunday",
    "preferred_shopping_window": {"start": "09:00", "end": "12:00"},

    # Default task duration in minutes when none is specified
    "default_durations": {
        "work":     60,
        "deep_work": 90,
        "chore":    30,
        "errand":   60,
        "health":   45,
        "personal": 30,
        "unknown":  30,
    },

    # Minimum gap between scheduled tasks (minutes)
    "buffer_minutes": 15,

    # Do not schedule anything before or after these times
    "hard_start": "06:00",
    "hard_end":   "21:00",
}
