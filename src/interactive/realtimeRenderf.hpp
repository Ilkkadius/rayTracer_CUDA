#ifndef REAL_TIME_RAY_TRACINGF_HPP
#define REAL_TIME_RAY_TRACINGF_HPP

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <memory>

#include "rayf.hpp"
#include "tracerf.hpp"
#include "rotationf.hpp"
#include "cameraf.hpp"
#include "kernelSet.hpp"

namespace cameraMove{

    void left(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0);

    void right(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0);

    void front(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0);

    void back(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0);

    void up(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0);

    void down(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed = 1.0);

    void rotateDirection(const sf::Vector2i& diff, Vector3D& direction, Vector3D& up, int width, int height, double speed = 1.0);

    void roll(const Vector3D& direction, Vector3D& up, double xangle);
}

namespace realtimeRender{
    void centerMouse(sf::Vector2i center, sf::Window& window);

    void detectKey(const sf::Event& event, Vector3D& eye, Vector3D& direction, Vector3D& up, bool& changes, int& speed);

    template <class Tptr>
    __host__ void simpleRender(sf::Uint8* pixels, Camera& cam, Tptr* targetHolder, WindowVectors* windowPtr, BackgroundColor** background, curandState* randState) {
        static int tx = 8, ty = 8;
        static dim3 blocks(divup(cam.width, tx), divup(cam.height, ty));
        static dim3 threads(tx, ty);

        *windowPtr = WindowVectors(cam.window);
        
        completeRender<<<blocks, threads>>>(pixels, cam.width, cam.height, cam.depth, cam.samples, targetHolder, background, windowPtr, randState);
        CHECK(cudaDeviceSynchronize());
    }

    template <class Tptr>
    __host__ void startCamera(Camera& cam, Tptr* targetHolder, BackgroundColor** background, curandState* randState, 
                            Vector3D& eye, Vector3D& direction, Vector3D& up) {
        cam.check();

        Camera camCpy = cam;
        camCpy.samples = 2;
        camCpy.width = camCpy.width/2;
        camCpy.height = camCpy.height/2;
        camCpy.depth = 3;

        sf::Texture texture;
        texture.create(camCpy.width, camCpy.height);

        sf::Uint8* pixels;
        CHECK(cudaMallocManaged(&pixels, camCpy.width * camCpy.height * 4));
        CHECK(cudaDeviceSynchronize());

        WindowVectors* windowPtr;
        CHECK(cudaMallocManaged(&windowPtr, sizeof(WindowVectors)));
        CHECK(cudaDeviceSynchronize());

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

                simpleRender(pixels, camCpy, targetHolder, windowPtr, background, randState);
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
        CHECK(cudaFree(windowPtr));
        CHECK(cudaDeviceSynchronize());
        cudaDeviceReset();
    }

}



#endif