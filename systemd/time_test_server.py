import socket
import time


def start_server(host='192.168.137.3', port=12345, expected_tests=100):
    """TCP延迟测试服务器端 - 连续接收"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"服务器启动在 {host}:{port}")

    try:
        print("等待客户端连接...")
        conn, addr = server_socket.accept()
        print(f"客户端连接: {addr}")

        # 连续接收100次测试数据
        for i in range(expected_tests):
            try:
                # 接收数据并记录接收时间
                data = conn.recv(1024)
                if not data:
                    break

                server_receive_time = time.time()

                # 解析客户端发送的时间戳
                timestamp = float(data.decode().split(':')[1])

                # 计算延迟
                latency = server_receive_time - timestamp
                print(f"测试 {i + 1:3d}: 客户端发送时间 {timestamp:.6f}, "
                      f"服务器接收时间 {server_receive_time:.6f}, "
                      f"网络延迟 {latency * 1000:.3f}ms")

                # 发送确认
                conn.send(b"RECEIVED")

            except Exception as e:
                print(f"处理第 {i + 1} 次测试数据时出错: {e}")
                break

        print("服务器接收测试完成")

    except KeyboardInterrupt:
        print("\n服务器关闭")
    finally:
        conn.close()
        server_socket.close()


if __name__ == "__main__":
    start_server(expected_tests=100)