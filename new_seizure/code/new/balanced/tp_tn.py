import keras.backend as K

def tp(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos) / (K.sum(y_pos) + 0.000000001)
    tn = K.sum(y_neg * y_pred_neg) / (K.sum(y_neg) + 0.000000001)
    return tp

def tn(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos) / (K.sum(y_pos) + 0.000000001)
    tn = K.sum(y_neg * y_pred_neg) / (K.sum(y_neg) + 0.000000001)
    return y_true  
