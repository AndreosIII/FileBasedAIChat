from typing import Dict, List, Tuple
import anthropic

from src.constants import AI_MARKER, HUMAN_MARKER

MAX_TOKENS = 4000


def anthropic_format_messages(conversation: List[Tuple]) -> List[Dict[str, str]]:
    """
    Форматирует историю диалога в список сообщений для Anthropic API.

    Args:
        conversation (list of tuple): История диалога, представленная списком кортежей (роль, содержание).

    Returns:
        list: Список словарей с отформатированными сообщениями для Anthropic API.
    """
    messages = []

    for role, content in conversation:
        if role == HUMAN_MARKER:
            anthropic_role = "user"
        elif role == AI_MARKER:
            anthropic_role = "assistant"
        messages.append({"role": anthropic_role, "content": content})

    return messages


def anthropic_send_request_to_model(
    model: str, api_key: str, api_base: str, behavior_description: str, temperature: float, messages: List
):
    """
    Отправляет запрос к модели Anthropic и получает потоковый ответ.

    Args:
        model (str): Идентификатор модели Anthropic, к которой будет выполнен запрос.
        api_key (str): API ключ для доступа к сервису Anthropic.
        api_base (str): Базовый URL API Anthropic.
        behavior_description (str): Описание поведения, которое будет использоваться как системное сообщение.
        temperature (float): Значение температуры для управления случайностью ответа.
        messages (list): Список сообщений в формате, требуемом Anthropic API.

    Yields:
        str: Ответы от модели по мере их генерации.
    """
    client = anthropic.Anthropic(api_key=api_key, base_url=api_base)

    try:
        response: anthropic.Stream = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,  # Используем константу вместо "magic number"
            temperature=temperature,
            system=behavior_description,
            messages=messages,
            stream=True,
        )

        for message in response:
            if message.type == "content_block_delta":
                yield message.delta.text
            elif message.type == "content_block_stop":
                break
    except Exception as e:
        print(f"Произошла ошибка при обращении к API Anthropic: {e}")


def communicate_with_anthropic(api_key: str, api_base: str, dialog_data: Dict):
    """
    Интегрирует взаимодействие с Anthropic API, используя данные из файла диалога.

    Args:
        api_key (str): API ключ для доступа к сервису Anthropic.
        api_base (str): Базовый URL API Anthropic.
        dialog_data (dict): Словарь с данными диалога, включая модель, температуру, описание поведения
        и историю диалога.

    Returns:
        Generator: Генератор, который выдает ответы модели по мере их получения.
    """
    model_name = dialog_data["model"]
    temperature = dialog_data["temperature"]
    behavior_description = dialog_data["behavior_description"]
    messages = anthropic_format_messages(dialog_data["conversation"])
    response = anthropic_send_request_to_model(
        model_name, api_key, api_base, behavior_description, temperature, messages
    )

    return response
