from typing import Dict, List, Tuple
import openai
from openai.error import OpenAIError, RateLimitError, InvalidRequestError, AuthenticationError
from src.constants import AI_MARKER, HUMAN_MARKER

DEFAULT_API_BASE = "https://api.openai.com"


def openai_format_messages(behavior_description: str, conversation: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """
    Форматирует описание поведения и историю диалога в список сообщений для OpenAI API.

    Args:
        behavior_description (str): Описание поведения, которое будет использоваться как системное сообщение.
        conversation (list of tuple): История диалога, представленная списком кортежей (роль, содержание).

    Returns:
        list: Список словарей с отформатированными сообщениями для OpenAI API.
    """
    messages = [{"role": "system", "content": behavior_description}]

    for role, content in conversation:
        if role == HUMAN_MARKER:
            openai_role = "user"
        elif role == AI_MARKER:
            openai_role = "assistant"
        messages.append({"role": openai_role, "content": content})

    return messages


def openai_send_request_to_model(
    model: str, api_key: str, temperature: float, messages: List, api_base: str = DEFAULT_API_BASE
):
    """
    Отправляет запрос к модели OpenAI и получает потоковый ответ.

    Args:
        model (str): Идентификатор модели OpenAI, к которой будет выполнен запрос.
        api_key (str): API ключ для доступа к сервису OpenAI.
        temperature (float): Значение температуры для управления случайностью ответа.
        messages (list): Список сообщений в формате, требуемом OpenAI API.
        api_base (str): Базовый URL API OpenAI.

    Yields:
        str: Ответы от модели по мере их генерации.

    Raises:
        RateLimitError: Превышен лимит запросов к API OpenAI.
        InvalidRequestError: Неверный запрос к API OpenAI.
        AuthenticationError: Ошибка аутентификации API ключа.
        OpenAIError: Прочие ошибки, связанные с API OpenAI.
        Exception: Неожиданные ошибки.
    """
    openai.api_key = api_key
    openai.api_base = api_base

    try:
        response = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature, stream=True)

        for chunk in response:
            if "content" in chunk["choices"][0]["delta"]:
                yield chunk["choices"][0]["delta"]["content"]

    except RateLimitError:
        print("Превышен лимит запросов к API OpenAI. Пожалуйста, попробуйте позже.")
    except InvalidRequestError as e:
        print(f"Неверный запрос к API OpenAI: {e}")
    except AuthenticationError:
        print("Ошибка аутентификации. Проверьте ваш API ключ.")
    except OpenAIError as e:
        print(f"Произошла ошибка при обращении к API OpenAI: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")


def communicate_with_openai(api_key: str, api_base: str, dialog_data: Dict):
    """
    Интегрирует взаимодействие с OpenAI API, используя данные из файла диалога.

    Args:
        api_key (str): API ключ для доступа к сервису OpenAI.
        api_base (str): Базовый URL API OpenAI.
        dialog_data (dict): Словарь с данными диалога, включая модель, температуру,
        описание поведения и историю диалога.

    Returns:
        Generator: Генератор, который выдает ответы модели по мере их получения.
    """
    model_name = dialog_data["model"]
    temperature = dialog_data["temperature"]
    messages = openai_format_messages(dialog_data["behavior_description"], dialog_data["conversation"])
    response = openai_send_request_to_model(model_name, api_key, temperature, messages, api_base)

    return response
