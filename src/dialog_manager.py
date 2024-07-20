import os
import json
import re
from typing import Dict, Generator, List, Tuple

from .constants import MODELS_CONFIG_PATH, HUMAN_MARKER, AI_MARKER

BEHAVIOR_CONFIG_PATH = "configs/behavior_templates.json"
DEFAULT_DIALOG_CONFIG_PATH = "configs/default_dialog_config.json"

MODEL_REGEX = r"^model:\s*(\S+)"
BEHAVIOR_REGEX = r"^behavior:\s*(\S+)"
CONVERSATION_REGEX = rf"^({HUMAN_MARKER}|{AI_MARKER}):\s*(.+?)(?=\n(?:{HUMAN_MARKER}|{AI_MARKER}):\s*|\Z)"


def load_json_config(file_path: str):
    """
    Загружает JSON файл и возвращает его содержимое в виде словаря.

    Args:
        file_path (str): Путь к JSON файлу, который нужно загрузить.

    Returns:
        Словарь с данными, загруженными из JSON файла.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def is_model_valid(model_name: str, models_config_path: str = MODELS_CONFIG_PATH) -> bool:
    """
    Проверяет, существует ли модель с указанным именем в конфигурационном файле моделей.

    Args:
        model_name (str): Имя модели для проверки.
        models_config_path (str): Путь к файлу конфигурации моделей.

    Returns:
        True, если модель существует, иначе False.
    """
    return model_name in load_json_config(models_config_path)


def is_behavior_valid(behavior_name: str, behavior_config_path: str = BEHAVIOR_CONFIG_PATH) -> bool:
    """
    Проверяет, существует ли шаблон поведения с указанным именем в конфигурационном файле шаблонов поведения.

    Args:
        behavior_name (str): Имя шаблона поведения для проверки.
        behavior_config_path (str): Путь к файлу конфигурации шаблонов поведения.

    Returns:
        True, если шаблон поведения существует, иначе False.
    """
    return behavior_name in load_json_config(behavior_config_path)


def create_default_dialog_file(file_path: str, default_config_path: str = DEFAULT_DIALOG_CONFIG_PATH) -> None:
    """
    Создает новый файл диалога с использованием стандартных настроек модели и поведения из конфигурационного файла.

    Args:
        file_path (str): Путь, по которому будет создан файл диалога.
        default_config_path (str): Путь к файлу со стандартной конфигурацией диалога.

    Raises:
        ValueError: Если указанная модель или шаблон поведения не найдены в соответствующих конфигурационных файлах.
    """
    default_config = load_json_config(default_config_path)
    model_name = default_config["default_model"]
    behavior_name = default_config["default_behavior"]

    if not is_model_valid(model_name):
        raise ValueError(f"Модель '{model_name}' не найдена в конфигурационном файле.")
    if not is_behavior_valid(behavior_name):
        raise ValueError(f"Шаблон поведения '{behavior_name}' не найден в файле шаблонов.")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"model: {model_name}\n")
        file.write(f"behavior: {behavior_name}\n\n")
        file.write(HUMAN_MARKER + ": ")


def check_or_create_dialog_file(file_path: str, default_config_path: str = DEFAULT_DIALOG_CONFIG_PATH) -> bool:
    """
    Проверяет существование файла диалога и создает его с настройками по умолчанию, если он не существует.

    Args:
        file_path: Путь к файлу диалога, который должен быть проверен или создан.
        default_config_path: Путь к файлу со стандартной конфигурацией диалога, используемой при создании нового файла.

    Returns:
        bool: True, если файл диалога уже существовал, иначе False.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if os.path.exists(file_path):
        return True
    create_default_dialog_file(file_path, default_config_path)
    return False


def parse_model_from_dialog(dialog_content: str):
    """
    Извлекает название модели из содержимого диалога.

    Args:
        dialog_content (str): Содержимое файла диалога в виде строки.

    Returns:
        Название модели, используемой в диалоге.

    Raises:
        ValueError: Если название модели отсутствует или модель не поддерживается.
    """
    model_match = re.search(MODEL_REGEX, dialog_content, re.MULTILINE)
    if model_match:
        model_name = model_match.group(1)
        if is_model_valid(model_name):
            return model_name
        raise ValueError(f"Указанная модель '{model_name}' не поддерживается.")
    raise ValueError("В файле диалога отсутствует указание модели.")


def parse_behavior_from_dialog(dialog_content: str):
    """
    Извлекает информацию о поведении из содержимого диалога.

    Args:
        dialog_content (str): Содержимое файла диалога в виде строки.

    Returns:
        tuple: Кортеж, содержащий описание поведения и температуру.

    Raises:
        ValueError: Если шаблон поведения отсутствует или не поддерживается.
    """
    behavior_match = re.search(BEHAVIOR_REGEX, dialog_content, re.MULTILINE)
    if behavior_match:
        behavior_name = behavior_match.group(1)
        if is_behavior_valid(behavior_name):
            behavior_config: Dict = load_json_config(BEHAVIOR_CONFIG_PATH)
            behavior_data: Dict = behavior_config.get(behavior_name)
            if behavior_data:
                description = behavior_data.get("description")
                temperature = behavior_data.get("temperature")
                return description, temperature
            raise ValueError(f"Данные поведения '{behavior_name}' не найдены в файле конфигурации.")
        raise ValueError(f"Указанное поведение '{behavior_name}' не поддерживается.")
    raise ValueError("В файле диалога отсутствует указание поведения.")


def parse_conversation_from_dialog(dialog_content: str) -> List[Tuple[str, str]]:
    """
    Извлекает высказывания участников диалога из содержимого файла.

    Args:
        dialog_content (str): Содержимое файла диалога в виде строки.

    Returns:
        Список кортежей, где каждый кортеж содержит маркер участника (Human или AI) и его высказывание.

    Raises:
        ValueError: Если последний запрос от пользователя пуст или отсутствует.
    """
    pattern = re.compile(CONVERSATION_REGEX, re.MULTILINE | re.DOTALL)
    conversation = []

    for match in pattern.finditer(dialog_content):
        speaker = match.group(1)
        statement = match.group(2).strip()
        conversation.append((speaker, statement))

    if not conversation or conversation[-1][0] != HUMAN_MARKER or not conversation[-1][1]:
        raise ValueError("Последний запрос от пользователя пуст или отсутствует.")

    return conversation


def parse_dialog_file(file_path: str):
    """
    Основная функция для парсинга файла диалога. Возвращает информацию о модели, поведении
    и высказываниях участников диалога.

    Args:
        file_path (str): Путь к файлу диалога, который нужно распарсить.

    Returns:
        dict: Словарь с ключами 'model', 'behavior_description', 'temperature', и 'conversation',
        содержащий информацию о модели, описании поведения, температуре и диалоге соответственно.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        dialog_content = file.read()

    model = parse_model_from_dialog(dialog_content)
    behavior_description, temperature = parse_behavior_from_dialog(dialog_content)
    conversation = parse_conversation_from_dialog(dialog_content)

    return {
        "model": model,
        "temperature": temperature,
        "behavior_description": behavior_description,
        "conversation": conversation,
    }


def append_ai_response_to_dialog(file_path: str, response_generator: Generator) -> None:
    """
    Добавляет ответ AI в файл диалога, следуя заданной разметке.

    Args:
        file_path (str): Путь к файлу диалога.
        response_generator (generator): Генератор, выдающий кусочки ответа AI.
    """
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(f"\n{AI_MARKER}: ")
        for chunk in response_generator:
            file.write(chunk)
            file.flush()  # Принудительно записываем данные из буфера в файл
        file.write(f"\n\n{HUMAN_MARKER}: ")
