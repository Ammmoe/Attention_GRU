def scale_per_agent(data, scaler, num_features_per_agent, fit=False, inverse=False):
    """
    Scale or inverse-scale per agent.
    
    Args:
        data: np.ndarray, shape (..., num_agents * per_agent_features)
        scaler: MinMaxScaler fitted on per-agent features
        num_features_per_agent: int, e.g. 3 for (x,y,z)
        fit: whether to fit the scaler
        inverse: if True, apply inverse_transform instead of transform
    """
    orig_shape = data.shape
    data_reshaped = data.reshape(-1, num_features_per_agent)  # collapse agents
    
    if fit:
        scaler.fit(data_reshaped)

    if inverse:
        data_scaled = scaler.inverse_transform(data_reshaped)
    else:
        data_scaled = scaler.transform(data_reshaped)

    return data_scaled.reshape(orig_shape)
