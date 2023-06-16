# Evaluate Model Performance
def model_evaluation(model, model_name):
    
    # print(model_performance)
    
    #Print Model Evaluations to the screen
    print(f"Train-Test Accuracy Scores:\n  Train: {round(model.score(X_train, y_train),5)} \n  Test: {round(model.score(X_test, y_test),5)}\n  Baseline: {round(dummy_accuracy,5)}\n---")
    print(f"\n Classification Report:\n{classification_report(y_test, model.predict(X_test), digits = 4)}")
    print(f"\n---\nBest Parameters: \n{model.best_params_}")
    
    # Plot and Save the Confusion Matrix
    preds = model.predict(X_test)
    plt.figure(figsize = (8,5))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap = 'YlOrBr', display_labels=['r/dating','r/datingoverthirty'])
    plt.title(f"Confusion Matrix: {model_name}")
    # plt.suptitle('Stop Words: English | Unigrams and Bigrams | Max Documents:90% | No Stem/Lem | LogisticRegression', y=0, fontsize = 9)
    plt.savefig(fname= f'./images/{model_name}_Confusion Matrix.png', bbox_inches = 'tight', dpi = 200)
    plt.show()
    
    #Append results of key metrics to 
    # pd.concat(model_performance_capture,
    model_performance = pd.DataFrame({
        'model_name' : model_name,
        'model' : model,
        'best_score_CV' : model.best_score_,
        'train_acuracy' : model.score(X_train, y_train),
        'test_accuracy' : model.score(X_test, y_test),
        'baseline_accuracy' : dummy_accuracy,
        'model_params' : [model.best_params_]
        })

    return model_performance