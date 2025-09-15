from sklearn.metrics import accuracy_score, classification_report

# Top 3 Model Terbaik
best_hps = tuner.get_best_hyperparameters(num_trials=3)
best_models = tuner.get_best_models(num_models=3)
print("\n == Top 3 Model Terbaik ==")
for i, (hp,model) in enumerate(zip(best_hps, best_models), start=1):
  # Evaluasi model pada data test
  loss, acc = model.evaluate(X_test, y_test, verbose=0)

  print(f'\nModel {i}:')
  print(f'Validation Accuracy: {acc:.4f}')
  print('Hyperparameters :')
  for param, value in hp.values.items():
    print(f'{param}:{value}')

best_models = tuner.get_best_models(num_models=1)[0]
best_models.summary()
loss, acc = best_models.evaluate(X_test, y_test)
print(f'Akurasi model terbaik pada data uji:{acc:.4f}')
print(f'Loss model terbaik pada data uji:{loss:.4f}')
