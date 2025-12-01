import torch


def output_change(output):
    data = output.clone()
    for i in range(len(data)):
        if data[i] >= 0.5:
            data[i] = 1.0
        elif data[i] < 0.5:
            data[i] = 0.0
        else:
            raise ValueError("invalid output")
    return data


def evaluateDNM(model, train_config, data):
    device = torch.device(train_config['device'])
    o_preds = []
    preds = []
    trues = []
    for i, (data_batch, label_batch) in enumerate(data):
        data_batch = data_batch.float().to(device)
        label_batch = label_batch.float().to(device)
        outputs = model(data_batch)
        outputs_p = output_change(outputs)
        o_preds.extend(outputs.detach().cpu().numpy())
        preds.extend(outputs_p.detach().cpu().numpy())
        trues.extend(label_batch.detach().cpu().numpy())
    return o_preds, preds, trues


def predict(model, model_config, train_config, data):
    train_type_fn = model_config['train_type_fn']
    with torch.no_grad():
        o_preds, preds, trues = eval(f'evaluate{train_type_fn}(model, train_config, data)')
    return o_preds, preds, trues
