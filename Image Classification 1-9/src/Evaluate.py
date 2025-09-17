from sklearn.metrics import accuracy_score, classification_report

# Top 3 Model Terbaik
best_hps = tuner.get_best_hyperparameters(num_trials=3)
best_models = tuner.get_best_models(num_models=3)
print("\n == Top 3 Model Terbaik ==")
for i, (hp,model) in enumerate(zip(best_hps, best_models), start=1):
  # Evaluasi model pada data test
  val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

  print(f'\nModel {i}:')
  print(f'Akurasi model terbaik pada data uji:{val_acc:.4f}')
  print(f'Loss model terbaik pada data uji:{val_loss:.4f}')
  print('Hyperparameters :')
  for param, value in hp.values.items():
    print(f'{param}:{value}')

best_models = tuner.get_best_models(num_models=1)[0]
best_models.summary()
