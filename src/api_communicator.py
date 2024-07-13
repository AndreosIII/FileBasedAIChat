import json
from .constants import MODELS_CONFIG_PATH
from .openai_api import communicate_with_openai
from .anthropic_api import communicate_with_anthropic

def get_api_credentials(model_name):
    """
    Получает учетные данные для API из конфигурационного файла моделей.

    Args:
        model_name (str): Имя модели, для которой необходимо получить учетные данные.

    Returns:
        tuple: Кортеж, содержащий api_key и api_base для указанной модели.
    """
    with open(MODELS_CONFIG_PATH, 'r', encoding='utf-8') as file:
        models = json.load(file)

    model_info = models[model_name]
    api_key = model_info.get('api_key')
    api_base = model_info.get('api_base')

    return api_key, api_base

def determine_model_type(model_name):
    """
    Определяет тип модели на основе имени модели.

    Args:
        model_name (str): Имя модели.

    Returns:
        str: Тип модели ('openai', 'anthropic', и т.д.)
    """
    if model_name.startswith(("gpt-", "llama", "mistral", "mixtral")):
        return "openai"
    elif model_name.startswith("claude-"):
        return "anthropic"
    else:
        raise ValueError(f"Неизвестный тип модели: {model_name}")

def communicate_with_ai_model(dialog_data):
    """
    Интегрирует взаимодействие с различными AI моделями, используя данные из файла диалога.

    Args:
        dialog_data (dict): Словарь с данными диалога, включая модель, температуру, описание поведения и историю диалога.

    Returns:
        Generator: Генератор, который выдает ответы модели по мере их получения.
    """
    model_name = dialog_data['model']
    model_type = determine_model_type(model_name)
    api_key, api_base = get_api_credentials(model_name)

    if model_type == "openai":
        return communicate_with_openai(api_key, api_base, dialog_data)
    elif model_type == "anthropic":
        return communicate_with_anthropic(api_key, api_base, dialog_data)
