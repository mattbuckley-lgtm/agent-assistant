---
name: word-limit-check
description: Check whether a given piece of text is within a specified word limit, and give feedback on how far over or under the limit it is.
when_to_use: The user wants to know if their text fits within a word limit, e.g. *"is my essay under 500 words?"*, *"does this fit in a 280-character tweet?"*, *"I have a 300 word limit, am I okay?"*
---
**Instructions:**
1. Extract the text and the target word limit from the user's request.
2. Call the `count_words` tool on the text.
3. Compare the result to the word limit and respond with one of:
- ✅ **Under limit:** *"Your text is **243 words**, which is **57 words under** your 300-word limit. You have room to expand!"*
- ⚠️ **At limit:** *"Your text is exactly **300 words** — perfect!"*
- ❌ **Over limit:** *"Your text is **342 words**, which is **42 words over** your 300-word limit. You'll need to trim it down."*