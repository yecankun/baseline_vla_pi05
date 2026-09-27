import socket
import time

def save_time(file_name, *args):
    root_path = "/home/xwj/桌面/project_2026/temp/data1"
    time_path = f"{root_path}/{file_name}.txt"
    args_string = ' '.join(map(str, args))
    write_in(time_path, args_string)

def write_in(file_name: str, data):
    with open(file_name, encoding="utf-8", mode="a") as file:
        file.write(str(data) + '\n')


def continuous_latency_test_single_connection(server_host='192.168.137.3', server_port=12345, test_count=100):
    """TCP延迟测试客户端 - 单次连接连续发送100次"""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_host, server_port))
    print(f"连接到服务器 {server_host}:{server_port}")

    latencies = []

    try:
        for i in range(test_count):
            # 记录发送时间
            send_time = time.time()
            message = f"TIMESTAMP:{send_time}"

            # 发送数据
            client_socket.send(message.encode())

            # 等待服务器确认
            response = client_socket.recv(1024)
            receive_confirm_time = time.time()

            # 计算往返时间
            round_trip_time = receive_confirm_time - send_time
            latencies.append(round_trip_time)

            print(f"测试 {i + 1:3d}: 往返时间 {round_trip_time * 1000:.3f}ms")

        # 统计结果
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)

            print(f"\n=== 测试完成 ({test_count} 次) ===")
            print(f"平均往返时间: {avg_latency * 1000:.3f}ms")
            print(f"最小往返时间: {min_latency * 1000:.3f}ms")
            print(f"最大往返时间: {max_latency * 1000:.3f}ms")
            print(f"总耗时: {sum(latencies) * 1000:.3f}ms")
        for t in latencies:
            save_time("tcp_time", t*500)

    except Exception as e:
        print(f"客户端错误: {e}")
    finally:
        client_socket.close()


if __name__ == "__main__":
    continuous_latency_test_single_connection(test_count=100)