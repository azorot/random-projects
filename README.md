# TradingviewLLM

This is my first GitHub-worthy project! I'm new to the coding scene, so if you use this code and experience issues, please let me know!

The way to run this code currently (I will add all pip packages in the future via requirements.txt.) 

```python grag.py```

Run in the terminal; it will have a link; click it, and it will open up the interface. You can generate code via query or just jump straight to training!

Note: There are a lot of kinks in the current version. redundant functions, unusable functions, etc... these will be fixed along the way. Perhaps wait for a better version.

Currently stuck on the training loop and getting the policy to update. But everything else works as intended! (Note: The selenium_handler script does not have successful code generation features implemented!)! This is due to the fact I'm using a very small model that is prone to errors, but this is also why I'm adding the RL technique to kind of mitigate this error-prone behavior.
