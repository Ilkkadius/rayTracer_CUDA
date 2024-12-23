#ifndef REAL_TIME_RAY_TRACINGF_HPP
#define REAL_TIME_RAY_TRACINGF_HPP

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <memory>

#include "geometria.hpp"
#include "rayf.hpp"
#include "tracerf.hpp"
#include "rotationf.hpp"
#include "cameraf.hpp"
#include "kernelSet.hpp"

#define CHECK_FUNC
static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(x) check(x, #x)

#define divup_FUNC

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

namespace cameraMove{

    void left(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0) {
        Vector3D r = unitVec(Cross(direction, up));
        eye += speed * 0.1 * r;
    }

    void right(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0) {
        Vector3D r = unitVec(Cross(direction, up));
        eye -= speed * 0.1 * r;
    }

    void front(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0) {
        eye += speed * 0.1 * unitVec(direction);
    }

    void back(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0) {
        eye -= speed * 0.1 * unitVec(direction);
    }

    void up(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0) {
        eye += speed * 0.1 * unitVec(up);
    }

    void down(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0) {
        eye -= speed * 0.1 * unitVec(up);
    }

    void rotateDirection(const sf::Vector2i& diff, Vector3D& direction, Vector3D& up, int width, int height, double speed = 1.0) {
        Vector3D unitDir = unitVec(direction);
        Vector3D unitUp = unitVec(up - Dot(up, unitDir) * unitDir);
        Vector3D unitRight = Cross(unitUp, unitDir);
        Vector3D change = diff.x * unitRight + diff.y * unitUp;
        double xangle = speed*double(diff.x) / width, yangle = speed*double(diff.y) / height;
        Matrix rotationx = generateRotation(xangle, up);
        direction = rotationx * direction; up = rotationx * up;
        Matrix rotationy = generateRotation(yangle, unitRight);
        direction = rotationy * direction; up = rotationy * up;
    }

    void roll(const Vector3D& direction, Vector3D& up, double xangle) {
        Matrix rotationx = generateRotation(xangle, direction);
        up = rotationx * up;
    }
}

namespace realtimeRender{
    void centerMouse(sf::Vector2i center, sf::Window& window) {
        sf::Mouse::setPosition(center, window);
    }

    void detectKey(const sf::Event& event, Vector3D& eye, Vector3D& direction, Vector3D& up, bool& changes, int& speed) {
        static double speedList[2] = {1.0, 20.0};
        speed = speed % 2;
        switch (event.key.code) {
            case sf::Keyboard::Left:
                // Left
                cameraMove::left(eye, direction, up, speedList[speed]);
                changes = true;
                break;
            case sf::Keyboard::A:
                // Left
                cameraMove::left(eye, direction, up, speedList[speed]);
                changes = true;
                break;
            case sf::Keyboard::Right:
                // Right
                cameraMove::right(eye, direction, up, speedList[speed]);
                changes = true;
                break;
            case sf::Keyboard::D:
                // Right
                cameraMove::right(eye, direction, up, speedList[speed]);
                changes = true;
                break;
            case sf::Keyboard::Down:
                // Backwards
                cameraMove::back(eye, direction, up, speedList[speed]);
                changes = true;
                break;
            case sf::Keyboard::S:
                // Backwards
                cameraMove::back(eye, direction, up, speedList[speed]);
                changes = true;
                break;
            case sf::Keyboard::Up:
                // Forwards
                cameraMove::front(eye, direction, up, speedList[speed]);
                changes = true;
                break;
            case sf::Keyboard::W:
                // Forwards
                cameraMove::front(eye, direction, up, speedList[speed]);
                changes = true;
                break;
            case sf::Keyboard::Space:
                // Up
                cameraMove::up(eye, direction, up, speedList[speed]);
                changes = true;
                break;
            case sf::Keyboard::LShift:
                // Down
                cameraMove::down(eye, direction, up, speedList[speed]);
                changes = true;
                break;
            case sf::Keyboard::Q:
                // Negative roll
                cameraMove::roll(direction, up, 0.01 * speedList[speed]);
                changes = true;
                break;
            case sf::Keyboard::E:
                // Positive roll
                cameraMove::roll(direction, up, -0.01 * speedList[speed]);
                changes = true;
                break;
        }
    }

    __host__ void simpleRender(sf::Uint8* pixels, Camera& cam, targetList** targetPtr, BackgroundColor** background, curandState* randState) {
        static int tx = 8, ty = 8;
        static dim3 blocks(divup(cam.width, tx), divup(cam.height, ty));
        static dim3 threads(tx, ty);

        WindowVectors* windowPtr;
        CHECK(cudaMallocManaged(&windowPtr, sizeof(WindowVectors)));
        CHECK(cudaDeviceSynchronize());
        
        *windowPtr = WindowVectors(cam.window); // Room for optimization here ..
        
        completeRender<<<blocks, threads>>>(pixels, cam.width, cam.height, cam.depth, cam.samples, targetPtr, background, windowPtr, randState);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaFree(windowPtr));
        CHECK(cudaDeviceSynchronize());
    }


    __host__ void startCamera(Camera& cam, targetList** targetPtr, BackgroundColor** background, curandState* randState, Vector3D& eye, Vector3D& direction, Vector3D& up) {
        cam.check();

        Camera camCpy = cam;
        camCpy.samples = 5;
        camCpy.width = camCpy.width/2;
        camCpy.height = camCpy.height/2;
        camCpy.depth = 3;

        sf::Texture texture;
        texture.create(camCpy.width, camCpy.height);

        sf::Uint8* pixels = new sf::Uint8[camCpy.width * camCpy.height * 4];
        CHECK(cudaMallocManaged(&pixels, camCpy.width * camCpy.height * 4));
        CHECK(cudaDeviceSynchronize());

        sf::Event event;
        sf::Sprite sprite;

        sf::RenderWindow window(sf::VideoMode(camCpy.width, camCpy.height), "RenderWindow_Frame0", sf::Style::Titlebar | sf::Style::Close);

        window.setMouseCursorVisible(true);
        window.setMouseCursorGrabbed(false);

        bool changes = true, mouseLocked = false;

        sf::Vector2i windowCenter(camCpy.width / 2, camCpy.height / 2), diff;
        int frame = 0;

        sf::Clock clock1;
        sf::Time time1;

        int speed = 0;

        while (window.isOpen()) {
            sf::Event event;
            while (window.pollEvent(event)) {

                if (event.type == sf::Event::Closed || sf::Keyboard::isKeyPressed(sf::Keyboard::Enter))
                    window.close();
                if (event.type == sf::Event::KeyPressed && window.hasFocus()) {
                    detectKey(event, camCpy.eye, camCpy.direction, camCpy.up, changes, speed);

                    if(event.key.code == sf::Keyboard::F) { // Fast mode on/off
                        speed++;
                        speed = speed % 2;
                        
                    }
                }
                if (event.type == sf::Event::MouseButtonPressed && window.hasFocus()) {
                    mouseLocked = true;
                    centerMouse(windowCenter, window);
                }
            } //* PollEvents above

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) { // End camera mode
                if(mouseLocked) centerMouse(windowCenter, window);
                mouseLocked = false;
            }

            if(window.hasFocus()) {

                if(sf::Keyboard::isKeyPressed(sf::Keyboard::V)) { // Save camera position
                    time1 = clock1.getElapsedTime();
                    if(time1.asMilliseconds() > 1200) {
                        eye = camCpy.eye; direction = camCpy.direction; up = camCpy.up;
                        std::cout << "\033[0;93mCurrent camera position:\033[0m" << std::endl;
                        std::cout << "eye: " << eye << std::endl;
                        std::cout << "direction: " << direction << std::endl;
                        std::cout << "up: " << up << std::endl;
                        clock1.restart();
                    }
                }

                if (mouseLocked) {
                    window.setMouseCursorVisible(false);
                    window.setMouseCursorGrabbed(true);
                    diff = sf::Mouse::getPosition(window) - windowCenter;
                    if (diff.x != 0 && diff.y != 0) {
                        cameraMove::rotateDirection(diff, camCpy.direction, camCpy.up, camCpy.width, camCpy.height);
                        changes = true;
                    }
                    centerMouse(windowCenter, window);
                } else {
                    window.setMouseCursorVisible(true);
                    window.setMouseCursorGrabbed(false);
                }
            }

            if (changes) {
                camCpy.check();

                simpleRender(pixels, camCpy, targetPtr, background, randState);
                frame++;
                window.setTitle("RenderWindow_Frame" + std::to_string(frame) + (speed == 1 ? "f" : ""));

                texture.update(pixels);
                window.clear();
                sprite.setTexture(texture);
                window.draw(sprite);
                window.display();

                changes = false;
            }
        }
        CHECK(cudaFree(pixels));
    }
}



#endif