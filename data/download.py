import kagglehub
kagglehub.login() # Il va vous demander votre nom d'utilisateur et votre clé API directement
path = kagglehub.competition_download('sia-predicting-short-form-video-popularity')
print(path)
