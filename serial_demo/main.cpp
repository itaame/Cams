#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <thread>
#include <termios.h>
#include <unistd.h>

int main() {
    const char* port = "/dev/ttyUSB0"; // adjust to your serial port
    int fd = open(port, O_WRONLY | O_NOCTTY);
    if (fd < 0) {
        std::perror("open");
        return 1;
    }

    termios tty{};
    if (tcgetattr(fd, &tty) != 0) {
        std::perror("tcgetattr");
        close(fd);
        return 1;
    }

    cfsetispeed(&tty, B115200);
    cfsetospeed(&tty, B115200);
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 5;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::perror("tcsetattr");
        close(fd);
        return 1;
    }

    const auto interval = std::chrono::microseconds(100); // 10 kHz
    int count = 0;
    while (true) {
        std::string msg = std::to_string(count++) + "\n";
        write(fd, msg.c_str(), msg.size());
        std::this_thread::sleep_for(interval);
    }

    close(fd);
    return 0;
}
