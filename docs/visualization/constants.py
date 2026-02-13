# Map technical metric names to readable names
METRIC_READABLE_MAP = {
    # Heart Rate (BPM)
    "HR_BPM_stats.min": "Batimento cardíaco mínimo (BPM)",
    "HR_BPM_stats.max": "Batimento cardíaco máximo (BPM)",
    "HR_BPM_stats.mean": "Média do batimento cardíaco (BPM)",
    "HR_BPM_stats.std": "Desvio padrão do batimento cardíaco (BPM)",

    # Heart Rate Ratio
    "HR_ratio_stats.min": "Rácio cardíaco mínima",
    "HR_ratio_stats.max": "Rácio cardíaco máxima",
    "HR_ratio_stats.mean": "Média do rácio cardíaco",
    "HR_ratio_stats.std": "Desvio padrão do rácio cardíaco",

    # Heart Rate Distributions
    "HR_distributions.Normal": "Percentagem do tempo com batimento normal",
    "HR_distributions.Ligeiramente elevado": "Percentagem do tempo com batimento ligeiramente elevado",
    "HR_distributions.Elevado": "Percentagem do tempo com batimento elevado",

    # Wrist Activities
    "WRIST_significant_rotation_percentage": "Percentagem de rotação significativa do pulso",
    "WRIST_significant_acceleration_percentage": "Percentagem de aceleração significativa do pulso",

    # Human Activities
    "HAR_distributions.Sentado": "Percentagem do tempo Sentado",
    "HAR_distributions.Andar": "Percentagem do tempo a andar",
    "HAR_distributions.De pé": "Percentagem do tempo de pé",

    "HAR_durations.Sentado_duration_sec": "Duração na posição sentada (s)",
    "HAR_durations.De pé_duration_sec": "Duração na posição de pé (s)",
    "HAR_durations.Andar_duration_sec": "Duração a andar (s)",

    "HAR_steps_num_steps": "Número total de passos",
    "HAR_steps_distance_walked_m": "Distância total percorrida a andar (m)",

    # Noise
    "Noise_distributions.Ruído baixo": "Percentagem de tempo em Ruído baixo",
    "Noise_distributions.Ruído incomodativo": "Percentagem de tempo em Ruído incomodativo",
    "Noise_distributions.Silencioso": "Percentagem de tempo Silencioso",
    "Noise_distributions.Ruído elevado": "Percentagem de tempo em Ruído elevado",

    "Noise_statistics.min": "Nível mínimo de ruído (dB)",
    "Noise_statistics.max": "Nível máximo de ruído (dB)",
    "Noise_statistics.mean": "Nível médio de ruído (dB)",
    "Noise_statistics.std": "Desvio padrão do ruído (dB)",

    "Noise_durations.Ruído baixo_duration_sec": "Duração em ruído baixo (s)",
    "Noise_durations.Ruído incomodativo_duration_sec": "Duração em ruído incomodativo (s)",
    "Noise_durations.Ruído elevado_duration_sec": "Duração em ruído elevado (s)",
    "Noise_durations.Silencioso_duration_sec": "Duração em ambiente silencioso (s)",

    # Posture
    "posture_ap_range": "Amplitude de oscilação antero-posterior",
    "posture_ml_range":"Amplitude de oscilação médio-lateral",
    "posture_ratio_range":"Relação AP/ML",
    "posture_total_sway_length":"Comprimento total do movimento",
    "posture_average_sway_velocity": "Velocidade média do movimento",
    "posture_sway_area_per_second": "Área de oscilação por segundo",
    "posture_95_confidence_ellipse_area": "Área elíptica de confiança 95%",

    # EMG
    "EMG_intensity.mean_percent_mvc": "EMG: Intensidade média em %MVC",
    "EMG_intensity.max_percent_mvc": "EMG: Intensidade máxima em %MVC",
    "EMG_intensity.min_percent_mvc": "EMG: Intensidade mínima em %MVC",
    "EMG_intensity.iemg_percent_seconds": "EMG: Percentagem IEMG",

    "EMG_apdf.full_p10": "EMG: Percentil 10 da APDF",
    "EMG_apdf.full_p50": "EMG: Percentil 50 da APDF",
    "EMG_apdf.full_p90": "EMG: Percentil 90 da APDF",

    "EMG_apdf.active_p10": "EMG: Percentil 10 da APDF (ativa)",
    "EMG_apdf.active_p50": "EMG: Percentil 50 da APDF (ativa)",
    "EMG_apdf.active_p90": "EMG: Percentil 90 da APDF (ativa)",

    "EMG_rest_recovery.rest_percent": "EMG: Percentagem de descanso",
    "EMG_rest_recovery.gap_frequency_per_minute": "EMG: Frequência de pausas por minuto",
    "EMG_rest_recovery.max_sustained_activity_s": "EMG: Duração máxima de atividade contínua (s)",
    "EMG_rest_recovery.gap_count": "EMG: Número de pausas detectadas",

    "EMG_relative_bins.below_usual_pct": "EMG: Tempo abaixo do nível habitual (%)",
    "EMG_relative_bins.typical_low_pct": "EMG: Tempo em nível típico baixo (%)",
    "EMG_relative_bins.typical_high_pct": "EMG: Tempo em nível típico alto (%)",
    "EMG_relative_bins.high_for_you_pct": "EMG: Tempo em nível alto pessoal (%)"


}