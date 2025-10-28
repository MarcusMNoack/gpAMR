#include <zmq.hpp>
#include <string>
#include <iostream>

int main() {
    zmq::context_t context(1);

    // Bind PULL socket to receive from Python
    zmq::socket_t pull_sock(context, ZMQ_PULL);
    pull_sock.bind("tcp://*:5555");

    // Connect PUSH socket to send to Python
    zmq::socket_t push_sock(context, ZMQ_PUSH);
    push_sock.connect("tcp://localhost:5556");

    // Example: send and receive
    std::string msg = "Hello from C++!";
    zmq::message_t zmq_msg(msg.data(), msg.size());
    push_sock.send(zmq_msg, zmq::send_flags::none);

    zmq::message_t recv_msg;
    pull_sock.recv(recv_msg, zmq::recv_flags::none);
    std::string received(static_cast<char*>(recv_msg.data()), recv_msg.size());
    std::cout << "Received from Python: " << received << std::endl;

    return 0;
}
