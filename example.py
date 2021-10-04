import os
import time
import numpy as np
import random
import copy

np.random.seed(1)


class Env:
    def __init__(self, render_speed=0.01, *args, **kwargs):
        self.render_speed = render_speed
        self.action_space = ['u', 'd', 'l', 'r', 'h']
        self.action_size = len(self.action_space)

        self.agent = {"pos": [0, 0], "hold": False, "hold_obj": -1}
        self.que = []

         # 환경 환경설정 부분
        self.map_size = [5, 6]
        self.type_count = 2

        self.base_map = self.create_base_map()
        self.fix_object_map = None
        # self.map = [] 출력할때만 필요해서 삭제

        self.counter = 0
        self.rewards = []
        self.objects = []
        self.goal = []

        self.rewards_switch = [0, 0]
        # 환경변수_불러오기
        self.env_var = kwargs['env_var']
        '''
        step_deduction: 스텝 소득 없을시 감점사항
        objects_point_add: object 가점 사항
        end_point_add: 종료지점 가점 사항
        '''

        self.checkPoint = False
        self.checkPoint_other = False

    def inline_scan(self):
        result = [i for i in range(self.map_size[0])]
        for i in range(self.map_size[0]):
            if sum(self.fix_object_map[1][i][0]):
                result.pop(result.index(i))
        return result

    # new methods
    def check_if_reward(self):
        check_list = dict()
        check_list['if_goal'] = False

        rewards = 0

        # checkpoint
        scan = False
        for i, obj in enumerate(self.objects):
            # if obj['pos'] == self.agent['pos'] and obj['type'] == 0:  # if조건 범위의 중요성
            if obj['pos'] == self.agent['pos']:
                if obj['type'] == self.que[0]:
                    scan = True
                    if self.rewards_switch[0] != 1:
                        self.rewards_switch[0] = 1
                        rewards += self.env_var['objects_point_add']  # 25
                    self.checkPoint = True
        if (not scan) and self.checkPoint:
            self.checkPoint = False
            if self.rewards_switch[0] != 0:
                self.rewards_switch[0] = 0
                rewards += self.env_var['objects_point_deduction']

        # checkpoint_other
        scan = False
        for i, obj in enumerate(self.objects):
            # if obj['pos'] == self.agent['pos'] and obj['type'] == 0:  # if조건 범위의 중요성
            if obj['pos'] == self.agent['pos']:
                if obj['type'] != self.que[0]:
                    scan = True
                    if self.rewards_switch[2] != 1:
                        self.rewards_switch[2] = 1
                        rewards += self.env_var['objects_other_point_add']  # 25
                    self.checkPoint_other = True
        if (not scan) and self.checkPoint_other:
            self.checkPoint_other = False
            if self.rewards_switch[2] != 0:
                self.rewards_switch[2] = 0
                rewards += self.env_var['objects_other_point_deduction']
        

        # hold point
        if self.agent['hold'] and self.objects[self.agent['hold_obj']]['type'] == self.que[0]:
            if self.rewards_switch[1] != 1:
                self.rewards_switch[1] = 1
                rewards += self.env_var['objects_hold_add']  # 35
        elif self.rewards_switch[1] != 0:
            self.rewards_switch[1] = 0
            rewards += self.env_var['objects_hold_deduction']

        # hold point_other
        if self.agent['hold'] and self.objects[self.agent['hold_obj']]['type'] != self.que[0]:
            if self.rewards_switch[3] != 1:
                self.rewards_switch[3] = 1
                rewards += self.env_var['objects_other_hold_add']  # 35
        elif self.rewards_switch[3] != 0:
            self.rewards_switch[3] = 0
            rewards += self.env_var['objects_other_hold_deduction']

        # inline

        inline_list_b = self.inline_scan()
        if len(inline_list_b)-len(self.inline_list) > 0:
            rewards += (len(inline_list_b) - len(self.inline_list))*self.env_var['inline_add']
        elif len(inline_list_b)-len(self.inline_list) == 0:
            rewards += self.env_var['inline_same']
        else:
            rewards += self.env_var['inline_deduction']

        # goal point
        if self.checkPoint and self.agent['pos'][1] == 5 and \
                self.agent['hold'] and self.objects[self.agent['hold_obj']]['type'] == self.que[0]:
            rewards += self.env_var['end_point_add']  # 40

            del self.objects[self.agent['hold_obj']]
            pos = self.agent['pos']
            self.fix_object_map[1][pos[0]][pos[1]] = [0 for i in range(self.type_count)]
            self.agent['hold'] = False
            self.agent['hold_obj'] = -1
            del self.que[0]

            self.checkPoint = False
            self.checkPoint_other = False
            self.rewards_switch = [0 for i in range(4)]

            check_list['if_goal'] = True

        # non target check
        for i in self.objects:
            if i['pos'][1] == 5 and i['type'] != self.que[0]:
                rewards += self.env_var['end_point_deduction']  # 40

        rewards += self.env_var['step_deduction']

        check_list['rewards'] = rewards

        return check_list

    def place_objects(self, count, work_range, except_position=[], self_place=[], types_count=[]):
        """# 가능한 y범위(세로) x범위(가로)+ 제외할 x,y 좌표 리스트

        # 랜덤 개수범위입력 -> 랜덤 배치
        # 수동배치도 가능"""

        result = []
        buffer = [i+1 for i in range(work_range[0]*work_range[1])]
        # 수동 추가
        if len(self_place):
            result += self_place
        # 추첨 제외
        for i in (self_place+except_position):
            if not(i[0]+1 > work_range[0] or i[1]+1 > work_range[1]):
                buffer.pop(buffer.index(work_range[0]*i[0]+i[1]+1))
        # 추첨
        if len(buffer) < count:
            count = len(buffer)
        result_buffer = random.sample(buffer, count)
        type_que = [0, 0]
        for i in result_buffer:
            if type_que[0]+1 != self.type_count:
                type_temp = type_que[0]
                type_que[1] += 1
                if types_count[type_que[0]] == type_que[1]:
                    type_que[0] += 1
                    type_que[1] = 0
            else:
                type_temp = self.type_count-1
            result.append({
                "pos": [(i - 1) // work_range[0], (i - 1) % work_range[0]],
                "type": type_temp,
                "is_fix": True
            })

        return result  # Example[{pos: [0, 1]}, {pos: [2, 15]}, {pos: [20, 13]}]

    def process_env(self):
        if (self.counter) % 200 == 0:
            entry_scan = self.inline_scan()

            self.objects.append({
                "pos": [random.sample(entry_scan, 1)[0], 0],
                "type": random.randint(0, 1),
                "is_fix": True
            })
            self.fix_object_map = self.create_fix_object_map()


        while len(self.que) < 3:
            type_count = [0 for i in range(self.type_count)]
            for i in self.objects:
                type_count[i['type']] += 1
            
            for i in self.que:
                type_count[i] -= 1
            
            # if sum(type_count):
            if sum(type_count) > 0:
                # if sum(type_count) > 0:
                    # temp = random.randint(1, sum(type_count))
                temp = random.randint(1, sum(type_count))
                for i, j in enumerate(type_count):
                    temp -= j
                    if temp <= 0:
                        break

                self.que.append(i)
            else:
                break



    def reset(self):
        self.counter = 0
        self.agent = {"pos": [0, 0], "hold": False, "hold_obj": -1}
        self.checkPoint = False
        self.checkPoint_other = False
        self.que = []
        self.rewards_switch = [0 for i in range(4)]
        """
        보상 지점 랜덤 설정 방법: 각 번호에 리스트를 만들어서 리스트에서 뽑고 리스트에서 제거
        """
        self.fix_object_map = copy.deepcopy(self.base_map)

        '''objects_count = random.randint(1, 8)
        self.objects = self.place_objects(
            objects_count, [5, 5], except_position=[], self_place=[],
            types_count=[objects_count//2]
        )'''
        self.objects = []

        # self.reset_reward()
        self.fix_object_map = self.create_fix_object_map()

        self.process_env()

        self.inline_list = self.inline_scan()
        check = self.check_if_reward()

        return self.get_state()


    def step(self, action):
        self.counter += 1
        # print(self.counter)

        self.process_env()

        self.inline_list = self.inline_scan()

        if action <= 3:
            self.agent['pos'] = self.move_agent(self.agent['pos'], action)
            self.hold_object_move()
        elif action == 4:
            self.hold_agent()

        check = self.check_if_reward()
        reward = check['rewards']
        done = False
        # done = sum(self.que) == -3
        if not done:
            entry_scan = self.inline_scan()
            if len(entry_scan) == 0:
                done = True
                reward += -1000


        s_ = self.get_state()

        return s_, reward/100, done

    # 반환할 상태에 맞게 배열 초기화
    def create_base_map(self):
        state_map = []  # [agent_pos, objects_map]

        # agnet_pos
        state_map.append([0, 0])  # agent position [y/map_size_y, x/map_size_x]

        # objects map
        state_map.append([])
        for i in range(self.map_size[0]):
            buffer = []
            for j in range(self.map_size[1]):
                buffer.append([0 for i in range(self.type_count)])
            state_map[1].append(buffer)

        return state_map

    def create_fix_object_map(self):
        temp_map = copy.deepcopy(self.base_map)

        # objects 배치
        for obj in self.objects:
            if obj['is_fix']:
                y = obj['pos'][0]  # 세로
                x = obj['pos'][1]  # 가로
                temp_map[1][y][x][obj['type']] = 1

        return temp_map

    def render(self):
        # state_map = copy.deepcopy(self.base_map)
        state_map = copy.deepcopy(self.fix_object_map)
        # agent 배치
        state_map[0][0] = self.agent['pos'][0] / (self.map_size[0]-1)
        state_map[0][1] = self.agent['pos'][1] / (self.map_size[1]-1)

        # objects 배치
        for obj in self.objects:
            if not obj['is_fix']:
                y = obj['pos'][0]  # 세로
                x = obj['pos'][1]  # 가로
                state_map[1][y][x][obj['type']] = 1

        state_map.append(self.checkPoint)
        state_map.append(self.agent['hold'])
        state_map.append([[0, 0] for i in range(3)])
        for i, j in enumerate(self.que):
            state_map[-1][i][j] = 1
        return state_map

    def get_state(self):  # map 에 속성값이 들어가 있는경우 반환할 상태로 변환처리
        # NOTE 속성으로 좌표를 찍어둔 다음 state 출력할때 위치를 포함하기
        states = self.render()

        return states

    # TODO def 환경 처리 함수 (ex - 매번 환경이 움직이는 경우 이것을 처리)

    def move_check(self, pos):  # pos base, 포지션 기반 검사
        if self.agent['hold']:
            result = sum(self.fix_object_map[1][pos[0]][pos[1]]) == 0
            return result
        else:
            return True

    def hold_agent(self):
        if self.agent['hold']:
            self.agent['hold_obj'] = -1
            self.agent['hold'] = False
        else:
            for i, obj in enumerate(self.objects):
                if self.agent['pos'] == obj['pos']:
                    self.agent['hold_obj'] = i
                    self.agent['hold'] = True

    def move_agent(self, pos, action):  # pos = [0, 0]
        before_pos = copy.deepcopy(pos)

        if action == 0:  # 상
            if pos[0] > 0:
                pos[0] -= 1
        elif action == 1:  # 하
            if pos[0] < self.map_size[0]-1:
                pos[0] += 1
        elif action == 2:  # 우
            if pos[1] < self.map_size[1] - 1:
                pos[1] += 1
        elif action == 3:  # 좌
            if pos[1] > 0:
                pos[1] -= 1

        if self.move_check(pos):
            return pos
        else:
            return before_pos

    def hold_object_move(self):
        if self.agent['hold']:
            pos = self.objects[self.agent['hold_obj']]['pos']
            self.fix_object_map[1][pos[0]][pos[1]] = [0 for i in range(self.type_count)]

            self.objects[self.agent['hold_obj']]['pos'] = copy.deepcopy(self.agent['pos'])
            pos = self.objects[self.agent['hold_obj']]['pos']
            self.fix_object_map[1][pos[0]][pos[1]][self.objects[self.agent['hold_obj']]['type']] = 1



    # 규칙에 맞게 시각화 구축
    def visualization(self, state):
        pos = [state[0][0]*(self.map_size[0]-1), state[0][1]*(self.map_size[1]-1)]
        print("que: {}".format(state[4]))
        print("hold:"+("●" if state[3] else "○"))
        for i, line in enumerate(state[1]):
            for j, point in enumerate(line):
                print("|{} {}".format(
                    ("●" if state[3] else "■") if pos == [i, j] else "□",
                    point.index(max(point)) if sum(point) != 0 else "-"
                ), end="")
            print("|")

if __name__ == "__main__":
    test_env = Env(env_var={
        "step_deduction": -0.0001,
        "objects_point_add": 25,
        "objects_point_deduction": -25,
        "objects_other_point_add": 1,
        "objects_other_point_deduction": -1,
        "objects_hold_add": 35,
        "objects_hold_deduction": -35,
        "objects_other_hold_add": 2,
        "objects_other_hold_deduction": -2,

        "end_point_add": 40,
        'end_point_deduction': -1,

    })
    a = test_env.reset()
    while False:
        u = input("code >>>")
        a, b, c = test_env.step(int(u))
        print(b, c)
    print(1)
    print(1)
    
