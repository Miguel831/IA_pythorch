print("Cargando el mejor modelo guardado...")
model.load_state_dict(torch.load('Modulo 3/best_model.pth'))  # cargar el modelo con mejor val_accuracy

    print("Evaluando en el conjunto de TEST...")
    test_accuracy, test_loss = evaluate(model, test_loader, criterion, device, epoch="final")

    print(f"RESULTADO FINAL EN TEST: Accuracy={test_accuracy:.2f}%, Loss={test_loss:.4f}")