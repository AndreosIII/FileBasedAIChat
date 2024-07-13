import argparse
from src.dialog_manager import parse_dialog_file, append_ai_response_to_dialog, check_or_create_dialog_file
from src.api_communicator import communicate_with_ai_model

def main():
    """
    Основная точка входа в приложение FileBasedAIChat.

    Эта функция выполняет следующие шаги:
    1. Разбирает аргументы командной строки, чтобы получить путь к файлу диалога.
    2. Проверяет существование файла диалога или создает новый файл с настройками по умолчанию.
    3. Читает и анализирует данные из файла диалога.
    4. Инициирует общение с выбранной моделью AI, используя API.
    5. Добавляет полученные от AI ответы в файл диалога.

    В случае прерывания процесса пользователем (Ctrl+C), выводит сообщение о прерывании.

    Аргументы командной строки:
    - file_path (str): Путь к файлу диалога, который будет обработан.

    При отсутствии файла диалога по указанному пути, будет создан новый файл с настройками
    по умолчанию. Если файл существует, программа приступит к обработке существующего диалога.
    """
    parser = argparse.ArgumentParser(description="Обработка файла диалога для FileBasedAIChat.")
    parser.add_argument('file_path', type=str, help="Путь к файлу диалога.")
    args = parser.parse_args()

    try:
        if check_or_create_dialog_file(args.file_path):
            dialog_data = parse_dialog_file(args.file_path)
            response = communicate_with_ai_model(dialog_data)
            append_ai_response_to_dialog(args.file_path, response)
        else:
            print("Файл диалога создан:", args.file_path)
    except KeyboardInterrupt:
        print("\nОтвет прерван пользователем.")

if __name__ == '__main__':
    main()
