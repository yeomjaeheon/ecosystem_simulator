#필요한 모듈들을 호출한다
import pygame, sys, random, dense_net
import numpy as np, copy
from pygame.locals import *

np.random.seed(111)

def f(l, n):
    s = 0
    norm = 0
    for i in range(0, len(l)):
        s += n ** i * l[i]
        norm += n ** i
    return s / norm

class art_life:
    def __init__(self, level, body, brain, spd, size):
        global fps
        self.energy = fps * 30
        self.level = level
        self.color = ((255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 128, 0), (128, 0, 128), (0, 0, 128), (128, 128, 128), (0, 0, 0))[self.level]
        self.body = body
        self.brain = brain
        if self.level == 1:
            a = f(self.brain.calcul([0.1 for i in range(0, 12)]), 2)
            c = f(self.brain.calcul([0.1 for i in range(0, 12)]), 12)
            self.color = (255 * a, 0, 255 * c)
        self.sight_limit = size * 100
        self.spd = spd
        self.size = size

def world_init(ecosystem, world_size, life_size, life_speed, brain_shape, sight):
    life = []
    for i in range(0, len(ecosystem)):
        for n in range(0, ecosystem[i]):
            life.append(art_life(i, pygame.Rect(random.randint(0, world_size[0] - life_size), random.randint(0, world_size[1] - life_size), life_size, life_size), dense_net.nn(8, 1, 4), life_speed, life_size))
    return life

def world_update(life, world_size):
    global fps
    life_tmp = [0 for i in range(0, len(life))]
    life_result = []
    for i in range(0, len(life)):
        for j in range(0, len(life)): #에너지가 부족해서 사망하는 경우를 처리함
            if life[i].energy <= 0:
                life_tmp[i] = None
            if i != j and life_tmp[i] != None and life_tmp[j] != None:
                if life[i].body.colliderect(life[j].body) and 0 < (life[i].level - life[j].level) <= 1: #먹히는 경우 처리
                    life[i].energy += life[j].energy
                    life_tmp[j] = None
    for i in range(0, len(life)):
        if life_tmp[i] != None:
            life_result.append(life[i])
    life_state = copy.deepcopy(life_result) #다음 상태를 계산하기 위해 현재의 상태를 저장해 둔다.
    for i in range(0, len(life_state)):
        sensor = [0.9 for i in range(0, 8)]
        for j in range(0, len(life_state)):
            if i != j:
                if pygame.Rect(life_state[i].body.right, life_state[i].body.top, life_state[i].sight_limit, life_state[i].size).colliderect(life_state[j].body): #우
                    if sensor[0] > (life_state[j].body.left - life_state[i].body.right) / life_state[i].sight_limit:
                        sensor[0] = (life_state[j].body.left - life_state[i].body.right) / life_state[i].sight_limit
                        sensor[1] = life_state[j].level
        for j in range(0, len(life_state)):
            if i != j:
                if pygame.Rect(life_state[i].body.left - life_state[i].sight_limit, life_state[i].body.top, life_state[i].sight_limit, life_state[i].size).colliderect(life_state[j].body): #좌
                    if sensor[2] > (life_state[i].body.left - life_state[j].body.right) / life_state[i].sight_limit:
                        sensor[2] = (life_state[i].body.left - life_state[j].body.right) / life_state[i].sight_limit
                        sensor[3] = life_state[j].level
        for j in range(0, len(life_state)):
            if i != j:
                if pygame.Rect(life_state[i].body.left, life_state[i].body.top - life_state[i].sight_limit, life_state[i].size, life_state[i].sight_limit).colliderect(life_state[j].body): #상
                    if sensor[4] > (life_state[j].body.bottom - life_state[i].body.top) / life_state[i].sight_limit:
                        sensor[4] = (life_state[j].body.bottom - life_state[i].body.top) / life_state[i].sight_limit
                        sensor[5] = life_state[j].level
        for j in range(0, len(life_state)):
            if i != j:
                if pygame.Rect(life_state[i].body.left, life_state[i].body.bottom, life_state[i].size, life_state[i].sight_limit).colliderect(life_state[j].body): #하
                    if sensor[6] > (life_state[i].body.bottom - life_state[j].body.top) / life_state[i].sight_limit:
                        sensor[6] = (life_state[i].body.bottom - life_state[j].body.top) / life_state[i].sight_limit
                        sensor[7] = life_state[j].level
        life_result[i] = life_update(world_size, life_state[i], sensor)
    new_life = []
    life_result_state = copy.deepcopy(life_result) #새로 생겨나는 생명체를 계산하기 위해 현재의 상태를 저장해 둔다.
    for i in range(0, len(life_result_state)):
        if life_result_state[i].energy > fps * 30:
            life_result[i].energy -= fps * 30
            br = copy.deepcopy(life_result[i].brain)
            br.mutate(random.random() * 0.1)
            new_life.append(art_life(life_result[i].level, pygame.Rect(random.randint(0, world_size[0] - life_size), random.randint(0, world_size[1] - life_size), life_result[i].size, life_result[i].size), br, life_speed, life_size))

    return life_result + new_life

def life_update(world_size, life_data, sensor):
    global energy_current
    global fps
    mode = 1
    life_data_result = life_data
    if life_data.level > 0:
        life_data_result.energy -= 1
        energy_current += 1
    if life_data.level > 0:
        output = life_data.brain.calcul(sensor)
        action = np.argmax(output)

        if action == 0:
            life_data_result.body.left -= life_data.spd
        elif action == 1:
            life_data_result.body.left += life_data.spd
        elif action == 2:
            life_data_result.body.top -= life_data.spd
        elif action == 3:
            life_data_result.body.top += life_data.spd

        if mode == 0:
            if life_data.body.left < 0:
                life_data_result.body.left += life_data.spd
            if life_data.body.right > world_size[0]:
                life_data_result.body.left -= life_data.spd
            if life_data.body.top < 0:
                life_data_result.body.top += life_data.spd
            if life_data.body.bottom > world_size[1]:
                life_data_result.body.bottom -= life_data.spd
        elif mode == 1:
            if life_data.body.left < 0:
                life_data_result.body.left += world_size[0]
            if life_data.body.right > world_size[0]:
                life_data_result.body.left -= world_size[0]
            if life_data.body.top < 0:
                life_data_result.body.top += world_size[1]
            if life_data.body.bottom > world_size[1]:
                life_data_result.body.bottom -= world_size[1]

    return life_data_result

def main():
    global timer
    global record_term
    global ecosystem_current
    global ecosystem
    global life
    global energy_current
    global fps
    global world_size
    #if timer >= fps * record_term:
        #timer = 0
        #for l in life:
            #ecosystem_current[l.level] += 1
        #print(ecosystem_current)
        #print(sum([i.energy for i in life]) + energy_current)
        #record.append(ecosystem_current)
        #ecosystem_current = [0 for i in ecosystem]
    #timer += 1
    life = world_update(life, world_size)
    if energy_current >= fps * 30:
        life.append(art_life(0, pygame.Rect(random.randint(0, world_size[0] - life_size), random.randint(0, world_size[1] - life_size), life_size, life_size), dense_net.nn(8, 1, 4), life_speed, life_size))
        energy_current -= fps * 30

#이 소스파일이 메인으로 동작될때만 실행
if __name__ == '__main__':
    #color = ((255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 128, 0), (128, 0, 128), (0, 0, 128), (128, 128, 128), (0, 0, 0))
    life_size = 16
    life_speed = 4
    sight = 50
    brain_shape = [12, 10, 10, 30, 9]
    pygame.init()
    #total_energy = 62
    record_term = 5 #5초
    record = []
    world_size = (1000, 750)
    fps_clock = pygame.time.Clock()
    fps = 30
    timer = 0
    ecosystem = (40, 10, 10) #각 종의 수 : (30, 30, 10)이 가장 좋은 결과가 나왔다.
    energy_current = 0
    ecosystem_current = [0 for i in ecosystem]
    life = world_init(ecosystem, world_size, life_size, life_speed, brain_shape, sight)
    display_surf = pygame.display.set_mode(world_size)
    pygame.display.set_caption('ecosystem_simulator')
    #게임 루프
    while True:
        display_surf.fill((255, 255, 255))
        for l in life:
            pygame.draw.rect(display_surf, l.color, l.body)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        main()
        pygame.display.update()
        fps_clock.tick(fps)
