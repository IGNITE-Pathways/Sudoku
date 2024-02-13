import cv2 as cv
import numpy as np
import imutils, gc
import pygame
import os, sys, time
import random
import pytesseract
from colorama import Fore, Back, Style

# Initial Size
WIDTH, HEIGHT = 1280, 720
systemExit=False
M = 9

# Global constants here
BLACK, WHITE, GREY = (0, 0, 0), (255, 255, 255), (50, 50, 50)
RED, GREEN, BLUE = (207, 0, 0), (0, 255, 0), (0, 100, 255)

KEYS=[pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, 
      pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]

class Sudoku:
    def __init__(self):
        self.buffer = 100
        self.margin = 50
        self.background = pygame.image.load('images/background.jpg').convert_alpha()
        self.game_selection_bg = pygame.image.load('images/game_selection.png').convert_alpha()
        self.camera = pygame.image.load('images/camera.png').convert_alpha()
        self.randomGame = pygame.image.load('images/random.png').convert_alpha()
        # Load million sudoku games from numpy array
        self.quizzes = np.load('data/sudoku_quizzes.npy') # shape = (1000000, 9, 9) 
        self.load_zones() # 9 zones
        # self.print_zones()
        self.sudoku_grid=[[0] * 9] * 9
        self.solution_grid=[[0] * 9] * 9
        # Default is randomly provided sudoku   
        self.byos = False 
        self.start_up_init()
        
    def start_up_init(self):
        self.background = pygame.transform.scale(self.background, (WIDTH, HEIGHT))
        self.cell_size = (WIDTH-2*self.margin)//3        
        self.welcomeFont = pygame.font.Font('font/IndianPoker.ttf', self.cell_size//4)
        self.font3 = pygame.font.Font('font/Quiltpatches-Ea2dr.otf', self.cell_size//2)

        self.welcomeText = self.welcomeFont.render("Welcome to Sudoku!", 1, BLACK)
        self.welcomeSize = self.welcomeFont.size("Welcome to Sudoku!")
        self.welcomeLoc = (WIDTH/2 - self.welcomeSize[0]/2, self.buffer)

        self.startButton = self.font3.render("Start", 1, BLACK)
        self.buttonSize =self.font3.size("Start")
        self.buttonLoc = (WIDTH/2 - self.buttonSize[0]/2, HEIGHT/2 - self.buttonSize[1]//2)
        self.buttonRect = pygame.Rect(self.buttonLoc, self.buttonSize)
        self.state = 0
        
    def game_selection_init(self):
        self.game_selection_bg = pygame.transform.scale(self.game_selection_bg, (WIDTH, HEIGHT))
        self.cell_size = (WIDTH-2*self.margin)//3
        self.font = pygame.font.Font('font/Chalkduster.ttf', self.cell_size//6)
        self.font2 = pygame.font.Font('font/IndianPoker.ttf', self.cell_size//10)
        self.sysFont = pygame.font.Font(pygame.font.get_default_font(), self.cell_size//6) 
        
        self.welcomeText = self.welcomeFont.render("Select a Game", 1, BLACK)
        self.welcomeSize = self.welcomeFont.size("Select a Game")
        self.welcomeLoc = (WIDTH/2 - self.welcomeSize[0]/2, self.buffer)

        self.cameraBtnSize = (WIDTH//8, WIDTH//8)
        self.camera = pygame.transform.scale(self.camera, self.cameraBtnSize)
        self.cameraBtnLoc = (WIDTH//2 - self.camera.get_width() - self.buffer//4, 
                             HEIGHT/2 - self.camera.get_height()//2)
        self.cameraBtnRect = pygame.Rect(self.cameraBtnLoc, self.cameraBtnSize)
        self.cameraSel0 = self.font2.render("0", 1, BLACK)
        self.cameraSel1 = self.font2.render("1", 1, BLACK)
        self.cameraSel2 = self.font2.render("2", 1, BLACK)
        self.cameraSelSize = self.font2.size("0")
        self.cameraSel0Loc = (self.cameraBtnLoc[0], self.cameraBtnLoc[1] + self.cameraBtnSize[1] + 10)
        self.cameraSel0Rect = pygame.Rect(self.cameraSel0Loc, self.cameraSelSize)
        self.cameraSel1Loc = (self.cameraBtnLoc[0] + self.cameraBtnSize[0]//2 - self.cameraSelSize[0]//2, 
                              self.cameraBtnLoc[1] + self.cameraBtnSize[1] + 10)
        self.cameraSel1Rect = pygame.Rect(self.cameraSel1Loc, self.cameraSelSize)
        self.cameraSel2Loc = (self.cameraBtnLoc[0] + self.cameraBtnSize[0] - self.cameraSelSize[0], 
                              self.cameraBtnLoc[1] + self.cameraBtnSize[1] + 10)
        self.cameraSel2Rect = pygame.Rect(self.cameraSel2Loc, self.cameraSelSize)

        self.cameraText1 = self.font2.render("Bring Your Own", 1, BLACK)
        self.cameraText1Size = self.font2.size("Bring Your Own")
        self.cameraText1Loc = (self.cameraBtnLoc[0] - self.cameraText1Size[0] - self.buffer//4, 
                               self.cameraBtnLoc[1] + self.cameraBtnSize[1]//2 - self.cameraText1Size[1] - 10)
        self.cameraText2 = self.font2.render("Sudoku (BYOS)", 1, BLACK)
        self.cameraText2Size = self.font2.size("Sudoku (BYOS)")
        self.cameraText2Loc = (self.cameraBtnLoc[0] - self.cameraText2Size[0] - self.buffer//4, 
                               self.cameraBtnLoc[1] + self.cameraBtnSize[1]//2 + 10)

        self.randomGameBtnSize = (WIDTH//8, WIDTH//8)
        self.randomGame = pygame.transform.scale(self.randomGame, self.randomGameBtnSize)
        self.randomGameBtnLoc = (WIDTH//2 + self.buffer//4, 
                                 HEIGHT/2 - self.randomGame.get_height()//2)
        self.randomGameBtnRect = pygame.Rect(self.randomGameBtnLoc, self.randomGameBtnSize)

        self.randomGameText1 = self.font2.render("Random", 1, BLACK)
        self.randomGameText1Size = self.font2.size("Random")
        self.randomGameText1Loc = (self.randomGameBtnLoc[0] + self.randomGameBtnSize[0] + self.buffer//4, 
                                   self.randomGameBtnLoc[1] + self.randomGameBtnSize[1]//2 - self.randomGameText1Size[1] - 10)
        self.randomGameText2 = self.font2.render("Sudoku", 1, BLACK)
        self.randomGameText2Size = self.font2.size("Sudoku")
        self.randomGameText2Loc = (self.randomGameBtnLoc[0] + self.randomGameBtnSize[0] + self.buffer//4, 
                               self.randomGameBtnLoc[1] + self.randomGameBtnSize[1]//2 + 10)

        if os.path.exists('out/quiz.jpg'):
            self.last_game_img = pygame.image.load('out/quiz.jpg').convert_alpha()
            self.replayBtnSize = (WIDTH//8, WIDTH//8)
            self.replayGame = pygame.transform.scale(self.last_game_img, self.replayBtnSize)
            self.replayBtnLoc = (WIDTH//2 - self.replayBtnSize[0]//2, self.randomGameBtnLoc[1] + self.randomGameBtnSize[1] + self.buffer)
            self.replayBtnRect = pygame.Rect(self.replayBtnLoc, self.replayBtnSize)
            self.replayText = self.font2.render("Play Again", 1, BLACK)
            self.replayTextSize = self.font2.size("Play Again")
            self.replayTextLoc = (WIDTH//2 - self.replayTextSize[0]//2, self.replayBtnLoc[1] + self.replayBtnSize[1] + 10)
        else:
            self.last_game_img = None

        self.selectedCamera = 0
        self.selected_cell_rect = None # user clicked on a cell (rect) on sudoku grid
        self.selected_zone = None # user clicked on a cell in the zone 
        self.selected_cell = None # user clicked on a cell on sudoku grid
        self.solved = False # Is sudoku solved?
        self.solve_to_the_end = False 
        self.sudoku_has_error = False
        self.sudoku_has_no_solution = False
        self.time_taken = 0
        self.start_time = 0
        self.solution_saved = False
        self.check_solution = False
        
    def play_init(self, replay=False):
        if not self.byos:
            self.random_select_init()
        else:
            print("BYOS")
            if not replay:
                #Start video capture 
                capture = cv.VideoCapture(self.selectedCamera)
                if not capture.isOpened():
                    print("Cannot open camera")
                    return
                s_box = None
                while True:
                    isTrue, frame = capture.read()
                
                    # if frame is read correctly ret is True
                    if not isTrue:
                        print("Can't receive frame (stream end?). Exiting ...")
                        break
                    
                    image = imutils.resize(frame, height = 700)
                    original_image = image.copy()
                    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                    x,y,w,h,c = self.get_sudoku_box(gray)
                    # draw guiding rectangle
                    cv.rectangle(image, (350, 100), (850, 600), (0,0,255), 2)
                    
                    if w >= 450:
                        sbox = original_image[y:y+h, x:x+w]
                        s_box = self.fix_sudoku_box(c,x,y,sbox)
                        break
                    if w != 0:
                        cv.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
                    cv.imshow('Scan Sudoku', image)
                    if cv.waitKey(20) & 0xFF==ord('d'):
                        break
                capture.release()
                cv.destroyAllWindows()
            else:
                #replay, load previous played sudoku
                s_box = cv.imread('out/quiz.jpg')

            # Size correction 
            if s_box is not None:
                self.sudoku_box = imutils.resize(s_box, height = 600) 
                sb_image = pygame.image.frombuffer(self.sudoku_box.tobytes(), 
                                                self.sudoku_box.shape[1::-1], "BGR")
                self.sudoku_box_image = pygame.transform.scale(sb_image, (WIDTH-100, HEIGHT-100))

                # Preprocess 
                warped600 = self.preprocess(self.sudoku_box)   
                grid_image = self.grids(warped600)
                # Finding Grid Points
                c2,bm,cnts = self.grid_points(grid_image,warped600)

                # Correcting the defects
                c2,num = self.get_digit(c2,bm,warped600,cnts)
                s_grid = self.sudoku_matrix(num)

                # self.board(s_grid)
                new_grid = s_grid.transpose()
                self.sudoku_grid = [[0]*9]*9
                # Convert grid
                for r in range(0,9,1):
                    self.sudoku_grid[r] = [0,0,0,0,0,0,0,0,0]
                    for c in range(0,9,1):
                        self.sudoku_grid[r][c] = (int(new_grid[r][c]),0)
                self.solution_grid = self.deep_copy(self.sudoku_grid)
                # if not self.solve_sudoku_from_pos(self.solution_grid, 0, 0, 0):
                #     self.sudoku_has_no_solution = True
                    
                self.print_sudoku(self.sudoku_grid)
            else:
                self.byos = False
                self.state = 1
                self.game_selection_init()
        
        self.hintBtn = self.font2.render(" Hint ", 1, WHITE)
        self.hintBtnSize =self.font2.size(" Hint ")
        self.hintBtnLoc = (WIDTH - self.margin - self.hintBtnSize[0] - 2, 10)
        self.hintBtnRect = pygame.Rect((self.hintBtnLoc[0],self.hintBtnLoc[1]-3), (self.hintBtnSize[0], self.hintBtnSize[1]+6))

        self.solveBtn = self.font2.render(" Solve ", 1, WHITE)
        self.solveBtnSize =self.font2.size(" Solve ")
        self.solveBtnLoc = (self.hintBtnLoc[0] - self.solveBtnSize[0] - 10, 10)
        self.solveBtnRect = pygame.Rect((self.solveBtnLoc[0], self.solveBtnLoc[1]-3), (self.solveBtnSize[0], self.solveBtnSize[1]+6))

        self.checkBtn = self.font2.render(" Check ", 1, WHITE)
        self.checkBtnSize =self.font2.size(" Check ")
        self.checkBtnLoc = (self.solveBtnLoc[0] - self.checkBtnSize[0] - 10, 10)
        self.checkBtnRect = pygame.Rect((self.checkBtnLoc[0], self.checkBtnLoc[1]-3), (self.checkBtnSize[0], self.checkBtnSize[1]+6))

    def main(self):
        if self.state == 0:
            self.screen_splash()
        elif self.state == 1:
            self.screen_game_selection()
        elif self.state == 2:
            self.screen_play()
            
    def screen_splash(self):
        global SCREEN, WIDTH, HEIGHT, systemExit
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                systemExit=True
                return
            elif event.type == pygame.VIDEORESIZE:
                SCREEN = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                WIDTH, HEIGHT = event.w, event.h
                self.start_up_init()
            # when the user clicks the start button, change to the playing state
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouseRect = pygame.Rect(event.pos, (1, 1))
                    if mouseRect.colliderect(self.buttonRect):
                        self.state = 1
                        WIDTH, HEIGHT = 1000, 1000
                        SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                        self.game_selection_init()
                        return

        #draw background
        SCREEN.blit(self.background, (0,0))
        #draw welcome text
        SCREEN.blit(self.welcomeText, self.welcomeLoc)
        #draw the start button
        SCREEN.blit(self.startButton, self.buttonLoc)
        pygame.display.flip()     
        
    def screen_game_selection(self):
        global SCREEN, WIDTH, HEIGHT, systemExit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                systemExit=True
                return
            if event.type == pygame.VIDEORESIZE:
                SCREEN = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                WIDTH = event.w
                HEIGHT = event.h
                self.game_selection_init()
            #when the user clicks the start button, change to the playing state
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouseRect = pygame.Rect(event.pos, (1,1))
                    if mouseRect.colliderect(self.cameraBtnRect):
                        self.state = 2
                        self.byos = True
                        self.selectedCamera=0
                        self.play_init()
                        return
                    elif mouseRect.colliderect(self.cameraSel0Rect):
                        self.state = 2
                        self.byos = True
                        self.selectedCamera=0
                        self.play_init()
                        return
                    elif mouseRect.colliderect(self.cameraSel1Rect):
                        self.state = 2
                        self.byos = True
                        self.selectedCamera=1
                        self.play_init()
                        return
                    elif mouseRect.colliderect(self.cameraSel2Rect):
                        self.state = 2
                        self.byos = True
                        self.selectedCamera=2
                        self.play_init()
                        return
                    elif mouseRect.colliderect(self.randomGameBtnRect):
                        self.state = 2
                        self.byos = False
                        self.play_init()
                        return
                    elif mouseRect.colliderect(self.replayBtnRect):
                        self.state = 2
                        self.byos = True
                        self.play_init(True)
                        return
                        
        #draw game selection screen background
        SCREEN.blit(self.game_selection_bg, (0,0))
        #draw welcome text // select game
        SCREEN.blit(self.welcomeText, self.welcomeLoc)
        #draw text - BYOS
        SCREEN.blit(self.cameraText1, self.cameraText1Loc)
        SCREEN.blit(self.cameraText2, self.cameraText2Loc)
        #draw camera
        SCREEN.blit(self.camera, self.cameraBtnLoc)
        SCREEN.blit(self.cameraSel0, self.cameraSel0Loc)
        SCREEN.blit(self.cameraSel1, self.cameraSel1Loc)
        SCREEN.blit(self.cameraSel2, self.cameraSel2Loc)
        
        #draw pick random game btn
        SCREEN.blit(self.randomGame, self.randomGameBtnLoc)
        #draw text - Random Selection 
        SCREEN.blit(self.randomGameText1, self.randomGameText1Loc)
        SCREEN.blit(self.randomGameText2, self.randomGameText2Loc)
        #Blit button outlines
        pygame.draw.rect(SCREEN, BLACK, self.cameraBtnRect, 2)
        pygame.draw.rect(SCREEN, BLACK, self.randomGameBtnRect, 2)
        if self.last_game_img is not None:
            SCREEN.blit(self.replayGame, self.replayBtnLoc)
            pygame.draw.rect(SCREEN, BLACK, self.replayBtnRect, 2)
            SCREEN.blit(self.replayText, self.replayTextLoc)
            
        pygame.display.flip()  
    
    def screen_play(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouseRect = pygame.Rect(event.pos, (1, 1))
                    if mouseRect.colliderect(self.hintBtnRect):
                        self.start_time = time.time() if self.start_time == 0 else self.start_time
                        self.next_iteration(self.sudoku_grid, True)  
                        # (x,y), num = self.get_hint() 
                        # self.sudoku_grid[x][y] = (num, 1)  
                    elif mouseRect.colliderect(self.solveBtnRect):
                        self.start_time = time.time() if self.start_time == 0 else self.start_time
                        if self.solve_sudoku_from_pos(self.sudoku_grid, 0, 0, 0):
                            self.time_taken = time.time() - self.start_time
                            print("After solve_sudoku")
                            self.print_sudoku(self.sudoku_grid)
                        else:
                            self.sudoku_has_no_solution = True
                        # self.solve_to_the_end = True      
                    elif mouseRect.colliderect(self.checkBtnRect):
                        self.check_solution = True
                    else:
                        self.selected_cell_rect, self.selected_zone, self.selected_cell, c_value = self.getCellRect(mouseRect)
                        if c_value[0] != 0 and c_value[1] == 0:
                            #Original value, can't change
                            self.selected_cell_rect = None           
                        if self.solved:
                            self.state = 1
                            self.game_selection_init()
            if self.selected_cell_rect is not None and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    x,y = self.get_grid_coordinates(self.selected_zone, self.selected_cell)
                    self.sudoku_grid[x][y] = (0,0)
                    self.sudoku_has_error = False
                    self.check_solution = False
                elif event.key in KEYS:
                    self.start_time = time.time() if self.start_time == 0 else self.start_time
                    num = int(event.unicode)
                    x,y = self.get_grid_coordinates(self.selected_zone, self.selected_cell)
                    self.sudoku_grid[x][y] = (num,1)
                    self.check_solution = False
        
        if self.solve_to_the_end:
            self.next_iteration(self.sudoku_grid, True)
            time.sleep(0.5)
            if self.solved or self.sudoku_has_error or self.sudoku_has_no_solution:
                print("Stopping solve loop")
                self.solve_to_the_end = False
            
        self.display_sudoku(self.sudoku_grid)
        self.solved = len(self.get_number_lists(self.sudoku_grid)[0]) == 0
        pygame.draw.rect(SCREEN, WHITE, self.hintBtnRect, 2)
        SCREEN.blit(self.hintBtn, self.hintBtnLoc)
        pygame.draw.rect(SCREEN, WHITE, self.solveBtnRect, 2)
        SCREEN.blit(self.solveBtn, self.solveBtnLoc)
        if self.sudoku_has_error:
            SCREEN.blit(self.font2.render("Error!", 1, RED), (self.margin,10))
        elif self.sudoku_has_no_solution:
            SCREEN.blit(self.font2.render("No Solution!", 1, RED), (self.margin,10))       
        elif self.solved:
            SCREEN.blit(self.font2.render("Time taken " + str(round(self.time_taken,1)) +"s", 1, WHITE), (self.margin,10))  
      
        if self.start_time != 0:
            pygame.draw.rect(SCREEN, WHITE, self.checkBtnRect, 2)
            SCREEN.blit(self.checkBtn, self.checkBtnLoc)
        
        pygame.display.flip()
        if self.solved and not self.solution_saved:
            self.save_sudoku_from_screen("out/solution.jpg")
            self.solution_saved = True
      
    def get_grid_coordinates(self, z, c):
        # return grid (x,y) given the zone and cell number
        return 3*((z-1)%3)+(c-1)%3, 3*((z-1)//3)+(c-1)//3
    
    def getCellRect(self,mouseRect):
        zone_clicked, cell_clicked, cell_rect, cell_value = None, None, None, (0,0)
        small_cell_size = self.cell_size//3
        for z in range(1,10):
            loc = (self.margin + self.cell_size*((z-1)%3), self.margin + self.cell_size*((z-1)//3))
            for c in range(1,10):
                cell_loc = (loc[0] + (small_cell_size)*((c-1)%3), loc[1] + small_cell_size*((c-1)//3))
                cell_rect = pygame.Rect(cell_loc, (small_cell_size, small_cell_size))
                if mouseRect.colliderect(cell_rect):
                    zone_clicked, cell_clicked = z, c
                    x,y = self.get_grid_coordinates(z,c)
                    cell_value = self.sudoku_grid[x][y]
                    print("Zone", zone_clicked, "Cell", cell_clicked, "Num", cell_value[0])
                    break
            if cell_clicked is not None:
                break
        return cell_rect, zone_clicked, cell_clicked, cell_value
    
    def random_select_init(self):
        print("Pick randomly")
        self.load_grid(random.randrange(0,1000000))
        self.display_sudoku(self.sudoku_grid)
        self.save_sudoku_from_screen("out/quiz.jpg")
        
    def fix_sudoku_box(self, cnt, x1, y1, image):
        perimeter=cv.arcLength(cnt,True)
        approx=cv.approxPolyDP(cnt, 0.05*perimeter,True)
        dst = np.zeros((4, 2), dtype = "float32")
        points = []
        for point in approx:
            x,y = point[0]
            points.append((x - x1,y - y1))
            print("Point", x,y)
        pts = np.array(points, dtype = "float32")
        rect = self.order_points(pts)
        print("Rect", rect)
        height, width = image.shape[:2]
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype = "float32")
        warped = self.four_point_transform(image, rect, dst, width, height)
        cv.imwrite('out/quiz.jpg', warped)
        return warped
    
    def order_points(self, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect

    def four_point_transform(self, image, rect, dst, width, height):
        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(image, M, (width, height))
        return warped
    
    def grids(self, warped2):
        img = np.zeros((600,600,3), np.uint8)
        frame = img
        img = cv.resize(frame,(610,610))
        
        # Finding Vertical lines
        for i in range(10):
            cv.line(img, (0,(img.shape[0]//9)*i),(img.shape[1],(img.shape[0]//9)*i), (255, 255, 255), 3, 1)
            cv.line(warped2, (0,(img.shape[0]//9)*i),(img.shape[1],(img.shape[0]//9)*i), (125, 0, 55), 3, 1)
        
        # Finding Horizontal Lines
        for j in range(10):
            cv.line(img, ((img.shape[1]//9)*j, 0), ((img.shape[1]//9)*j, img.shape[0]), (255, 255, 255), 3, 1)
            cv.line(warped2, ((img.shape[1]//9)*j, 0), ((img.shape[1]//9)*j, img.shape[0]), (125, 0, 55), 3, 1)
    
        return img
    
    def grid_points(self, grid_image,warped2):
        # Finding out the intersection pts to get the grids
        img1 = grid_image.copy()
        kernelx = cv.getStructuringElement(cv.MORPH_RECT,(2,10))

        dx = cv.Sobel(grid_image,cv.CV_16S,1,0)
        dx = cv.convertScaleAbs(dx)
        c = cv.normalize(dx,dx,0,255,cv.NORM_MINMAX)
        c = cv.morphologyEx(c,cv.MORPH_DILATE,kernelx,iterations = 1)
        cy = cv.cvtColor(c,cv.COLOR_BGR2GRAY)
        closex = cv.morphologyEx(cy,cv.MORPH_DILATE,kernelx,iterations = 1)

        kernely = cv.getStructuringElement(cv.MORPH_RECT,(10,2))
        dy = cv.Sobel(grid_image,cv.CV_16S,0,2)
        dy = cv.convertScaleAbs(dy)
        c = cv.normalize(dy,dy,0,255,cv.NORM_MINMAX)
        c = cv.morphologyEx(c,cv.MORPH_DILATE,kernely,iterations = 1)
        cy = cv.cvtColor(c,cv.COLOR_BGR2GRAY)
        closey = cv.morphologyEx(cy,cv.MORPH_DILATE,kernelx,iterations = 1)

        res = cv.bitwise_and(closex,closey)
        ret, thresh = cv.threshold(res,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        kernel = np.ones((6,6),np.uint8)
        
        # Perform morphology
        se = np.ones((8,8), dtype='uint8')
        image_close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, se)
        image_close = cv.morphologyEx(image_close, cv.MORPH_OPEN, kernel)

        contour, hier = cv.findContours(image_close,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contour, key=cv.contourArea, reverse=True)[:100]
        centroids = []
    
        for cnt in cnts:
            mom = cv.moments(cnt)
            (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
            cv.circle(img1,(x,y),4,(0,255,0),-1)
            cv.circle(warped2,(x,y),4,(0,255,0),-1)
            centroids.append((x,y))

        Points = np.array(centroids,dtype = np.float32)
        c = Points.reshape((100,2))
        c2 = c[np.argsort(c[:,1])]

        b = np.vstack([c2[i*10:(i+1)*10][np.argsort(c2[i*10:(i+1)*10,0])] for i in range(10)])
        bm = b.reshape((10,10,2))
        # cv.imshow("Grid Dots", warped2)
        return c2,bm,cnts
    
    def get_digit(self, c2,bm,warped1,cnts):
        # To get the digit at the particular cell
        num = []        
        for i in range(0,9):
            for j in range(0,9):
                x1,y1 = bm[i][j] # bm[0] row1 
                x2,y2 = bm[i+1][j+1]
                
                crop = warped1[int(x1):int(x2),int(y1):int(y2)]
                crop = imutils.resize(crop, height=69,width=69)
                c2 = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
                c2 = cv.adaptiveThreshold(c2,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,11,2)
                c2= cv.copyMakeBorder(c2,5,5,5,5,cv.BORDER_CONSTANT,value=(0,0,0))
                no = 0
                shape=c2.shape
                w,h=shape[1], shape[0]
                c2 = c2[14:70,15:62]
                contour, hier = cv.findContours(c2,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
                if cnts is not None:
                    cnts = sorted(contour, key=cv.contourArea,reverse=True)[:1]

                for cnt in cnts:
                    x,y,w,h = cv.boundingRect(cnt)
                    aspect_ratio = w/h
                    area = cv.contourArea(cnt)
                    # print("Area", area, "Shape", cnt.shape[0], "Aspect", aspect_ratio)
                    if area>70 and cnt.shape[0]>10 and aspect_ratio>0.2 and aspect_ratio<=0.9: 
                        c2 = self.find_largest_feature(c2)
                        contour, hier = cv.findContours (c2,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
                        cnts = sorted(contour, key=cv.contourArea,reverse=True)[:1]
                        for cnt in cnts:
                            rect = cv.boundingRect(cnt)
                            c2 = c2[rect[1]:rect[3]+rect[1],rect[0]:rect[2]+rect[0]]
                            c2= cv.copyMakeBorder(c2,5,5,5,5,cv.BORDER_CONSTANT,value=(0,0,0))
                            # self.show_image("image_to_num", c2)
                        no = self.image_to_num(c2)
                        # print("Found Number", no)
                num.append(no)
        return c2, num

    def find_largest_feature(self, inp_img, scan_tl=None, scan_br=None):
        # Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
        # connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
        img = inp_img.copy()  # Copy the image, leaving the original untouched
        height, width = img.shape[:2]

        max_area = 0
        seed_point = (None, None)

        if scan_tl is None:
            scan_tl = [0, 0]

        if scan_br is None:
            scan_br = [width, height]

        # Loop through the image
        for x in range(scan_tl[0], scan_br[0]):
            for y in range(scan_tl[1], scan_br[1]):
                # Only operate on light or white squares
                if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
                    area = cv.floodFill(img, None, (x, y), 64)
                    if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                        max_area = area[0]
                        seed_point = (x, y)

        # Colour everything grey (compensates for features outside of our middle scanning range
        for x in range(width):
            for y in range(height):
                if img.item(y, x) == 255 and x < width and y < height:
                    cv.floodFill(img, None, (x, y), 64)

        mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

        # Highlight the main feature
        if all([p is not None for p in seed_point]):
            cv.floodFill(img, mask, seed_point, 255)

        for x in range(width):
            for y in range(height):
                if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                    cv.floodFill(img, mask, (x, y), 0)
                    
        return img

    def image_to_num(self, c2):     
        img = 255-c2
        text = pytesseract.image_to_string(img, lang="eng",config='--psm 6 --oem 3') #builder=builder)
        return str(list(text)[0])

    def sudoku_matrix(self, num):
        # creating matrix and filling numbers exist in the orig image 
        c = 0
        grid = np.empty((9, 9))
        for i in range(9):
            for j in range(9):
                if type(num[c]) == int or type(num[c]) == float:
                    grid[i][j] = int(num[c])
                elif type(num[c]) == str and str(num[c]).isnumeric():
                    grid[i][j] = int(num[c])
                else:
                    grid[i][j] = 0
                c += 1
        grid = np.transpose(grid)
        return grid

    def board(self, arr):
        for i in range(9):
        
            if i%3==0 :
                    print("+",end="")
                    print("-------+"*3)
                    
            for j in range(9):
                if j%3 ==0 :
                    print("",end="| ")
                print(int(arr[i][j]),end=" ")
        
            print("",end="|")       
            print()
        
        print("+",end="")
        print("-------+"*3)
        return arr  
    
    def get_sudoku_box(self, image):        
        thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        # cv.imshow("thresh", thresh)
        cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        x,y,w,h = 0,0,0,0
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]
        found = False
        contour_found = None
        for c in cnts:
            area = cv.contourArea(c)
            x,y,w,h = cv.boundingRect(c)
            aspect_ratio = w/h
            if area>20000 and c.shape[0]<100 and aspect_ratio > 0.8 and aspect_ratio < 1.2: 
                # cv.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
                # cropped_box = image[y:y+h, x:x+w]
                found = True
                contour_found = c
                break
        if found:
            return x,y,w,h,contour_found
        else:
            return 0,0,0,0,contour_found
    
    def preprocess(self, image):
        warped500 = imutils.resize(image, height = 600)     
        warped600 = cv.resize(warped500,(610,610))    
        return warped600

    def print_number_lists(self, nl):
        for i in range(1,10): 
            print(i,"(",len(nl[i]),")"," -->", nl[i])
                
    def check_row_column_zone(self,grid, row, col, num):
        # Check occurance of num in row
        for y in range(9):
            if grid[row][y][0] == num:
                return False
                
        # Check occurance of num in column
        for x in range(9):
            if grid[x][col][0] == num:
                return False
    
        # Check occurance of num in zone
        startRow = row - row % 3
        startCol = col - col % 3
        for i in range(3):
            for j in range(3):
                if grid[i + startRow][j + startCol][0] == num:
                    return False
        return True

    def solve_sudoku_from_pos(self, grid, row, col, stack):
        # print("."*stack)  
        self.display_sudoku(grid)
        pygame.display.flip()

        if (row == M - 1 and col == M):
            return True
        if col == M:
            row += 1
            col = 0
        if grid[row][col][0] > 0:
            return self.solve_sudoku_from_pos(grid, row, col + 1, stack + 1)
        
        for num in range(1, M + 1, 1): 
            if self.check_row_column_zone(grid, row, col, num):
                grid[row][col] = (num,1)
                if self.solve_sudoku_from_pos(grid, row, col + 1, stack + 1):
                    return True
            grid[row][col] = (0,0)
        return False

    def save_sudoku_from_screen(self, file):
        # Capture Pygame Screen with opencv ndArray
        capture = pygame.surfarray.pixels3d(SCREEN)
        capture = capture.transpose([1, 0, 2])
        capture_bgr = cv.cvtColor(capture, cv.COLOR_RGB2BGR)
        # Crop out the margin
        s_box = capture_bgr[self.margin:WIDTH-self.margin,self.margin:HEIGHT-self.margin]   
        s_box = imutils.resize(s_box, height = 600)   
        cv.imwrite(file, s_box)
        
    def deep_copy(self, grid):
        copy = [0]*9
        for x in range(0,9,1):
            copy[x] = [0,0,0,0,0,0,0,0,0]
            for y in range(0,9,1):
                copy[x][y] = grid[x][y]
        return copy
        
    def next_iteration(self, s_grid, populate_one=False):
        #Find next set of values
        number_lists = self.get_number_lists(s_grid)
        possible_solution = self.find_possible_solution_number_lists(s_grid, number_lists)
        s_grid = self.populate_unique_pos(possible_solution, s_grid, populate_one)
    
    def get_number_lists(self, g):
        # key is number (1-9) and value is the list of (x,y) of number positions
        number_lists={1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],0:[]}
        for y in range(0,9,1):
            for x in range(0,9,1): 
                number_lists[g[x][y][0]].append((x,y))
        return number_lists
            
    def is_number_present_in_zone(self, num, s_grid, z):
        found = False
        for (x,y) in self.zones[z]:
            if s_grid[x][y][0] == num:
                found = True
                break
        return found
        
    def find_possible_solution_number_lists(self, s_grid, number_lists):
        possible_solution={1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],0:[]}
        for i in range(1,10): 
            possible_solution[i] = self.find_possible_solution_for_number_in_list(i, s_grid, number_lists[i])
        return possible_solution
    
    def find_possible_solution_for_number_in_list(self, i, s_grid, nl):
        ps = []
        xvalues=[0,1,2,3,4,5,6,7,8]
        yvalues=[0,1,2,3,4,5,6,7,8]
        
        # print(i, "incoming number_list", nl)
        for num in nl:
            if num[0] in xvalues:
                xvalues.remove(num[0])
            if num[1] in yvalues:
                yvalues.remove(num[1])
            
        # print(i, "find_possible_solution xvalues yvalues", xvalues, yvalues)
        # Iterate to populate all possible values
        for x in xvalues:
            for y in yvalues:
                if s_grid[x][y][0] == 0:
                    ps.append((x,y))
        # print(i, "find_possible_solution before removing for zone", ps)
        
        # check z grid, remove if box already contains num
        for z in range(1,10): 
            if self.is_number_present_in_zone(i, s_grid, z):
                for (x,y) in self.zones[z]:
                    if (x,y) in ps:
                        # print("zone",z, "removing",(x,y), "for number", i, "from", ps)
                        ps.remove((x,y))
        return ps
                        
    def get_possible_positions_for_a_number_in_a_zone(self, num, z, s):
        count = 0
        position_list = []
        for (x,y) in self.zones[z]:
            # iterate through every cell in given zone (z)
            if (x,y) in s[num]:
                # check solution for given number (num)
                if (x,y) not in position_list:
                    count = count+1
                    position_list.append((x,y))
        return count, position_list
            
    def get_missing_numbers_in_a_zone(self, z, s_grid):
        missng_numbers_in_zone = [1,2,3,4,5,6,7,8,9]
        found_numbers_pos = []
        for (x,y) in self.zones[z]:
            # iterate through every cell in given zone (z)
            if s_grid[x][y][0] != 0 and s_grid[x][y][0] in missng_numbers_in_zone:
                # removing existing numbers from missing list
                missng_numbers_in_zone.remove(s_grid[x][y][0])
                found_numbers_pos.append((x,y))
        zone = self.zones[z].copy()
        missng_numbers_positions = [i for i in zone if i not in found_numbers_pos]
        return missng_numbers_in_zone, missng_numbers_positions
    
    def get_missing_numbers_in_a_row(self, row_num, s_grid):
        # print("get_missing_numbers_in_a_row", row_num, s_grid)
        missng_numbers_in_row = [1,2,3,4,5,6,7,8,9]
        missng_numbers_positions = []
        # iterate through every cell in given row (row_num)
        for x in range(0,9,1):
            # removing existing numbers from missing list
            if s_grid[x][row_num][0] != 0 and s_grid[x][row_num][0] in missng_numbers_in_row: ###########################################
                missng_numbers_in_row.remove(s_grid[x][row_num][0])
            else:
                #pos (x,row_num) has 0
                missng_numbers_positions.append((x,row_num))            
        return missng_numbers_in_row, missng_numbers_positions
    
    def get_missing_numbers_in_a_column(self, col_num, s_grid):
        missng_numbers_in_col = [1,2,3,4,5,6,7,8,9]
        missng_numbers_positions = []
        # iterate through every cell in given column (col_num)
        for y in range(0,9,1):
            # removing existing numbers from missing list
            if s_grid[col_num][y][0] != 0 and s_grid[col_num][y][0] in missng_numbers_in_col:
                missng_numbers_in_col.remove(s_grid[col_num][y][0])
            else:
                #pos (col_num,0) has 0
                missng_numbers_positions.append((col_num,y))            
        return missng_numbers_in_col, missng_numbers_positions
    
    def populate_unique_pos(self, possible_s, s_grid, populate_one=False):
        # iterate every zone (and row and column)
        populated = False
        for z in range(1,10): 
            # check every number in every zone 
            for num in range(1,10): 
                count, position_list = self.get_possible_positions_for_a_number_in_a_zone(num, z, possible_s)
                if count == 1:
                    # assign if only single possibility
                    s_grid[position_list[0][0]][position_list[0][1]] = (num,1)
                    populated = True
                    if populate_one:
                        break  #inner loop                  
            if populated and populate_one:
                break #outer loop
            else:
                missng_numbers_in_zone, missng_numbers_positions = self.get_missing_numbers_in_a_zone(z, s_grid)
                if len(missng_numbers_in_zone) == 1:
                    s_grid[missng_numbers_positions[0][0]][missng_numbers_positions[0][1]] = (missng_numbers_in_zone[0],1)
                    populated = True
                if populated and populate_one:
                    break #outer loop
                else:    
                    missng_numbers_in_row, missng_numbers_positions = self.get_missing_numbers_in_a_row(z-1, s_grid)
                    if len(missng_numbers_in_row) == 1:
                        s_grid[missng_numbers_positions[0][0]][missng_numbers_positions[0][1]] = (missng_numbers_in_row[0],1)
                        populated = True
                    if populated and populate_one:
                        break #outer loop
                    else:    
                        missng_numbers_in_column, missng_numbers_positions = self.get_missing_numbers_in_a_column(z-1, s_grid)
                        if len(missng_numbers_in_column) == 1:
                            s_grid[missng_numbers_positions[0][0]][missng_numbers_positions[0][1]] = (missng_numbers_in_column[0],1)
                            populated = True
            if populated and populate_one:
                break #outer loop
        return s_grid
                            
    def load_grid(self, random_num):
        for r in range(0,9,1):
            self.sudoku_grid[r] = [0,0,0,0,0,0,0,0,0]
            for c in range(0,9,1):
                self.sudoku_grid[r][c] = (self.quizzes[random_num][r][c],0)
        self.solution_grid = self.deep_copy(self.sudoku_grid)
        # if not self.solve_sudoku_from_pos(self.solution_grid, 0, 0, 0):
        #     self.sudoku_has_no_solution = True
    
    def get_hint(self):
        x = random.randrange(0,9)
        y = random.randrange(0,9)
        if self.sudoku_grid[x][y][0] == 0:
            return (x,y), self.solution_grid[x][y][0]
        else:
            return self.get_hint()
        
    def get_zone_number(self, x,y): 
        return (x)//3 + (3 * ((y)//3)) + 1
    
    def print_zones(self):
        for z in range(1,10): 
            print("Zone", z, "::", self.zones[z])
        
    def load_zones(self):
        self.zones={1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],0:[]}
        for x in range(0,9,1): 
            for y in range(0,9,1):
                self.zones[self.get_zone_number(x,y)].append((x,y))
                
    def display_zone(self, zone, cs, margin=(50,50), thinkness=2):
        loc = (margin[0] + cs*((zone-1)%3), margin[1] + cs*((zone-1)//3))
        pygame.draw.rect(SCREEN, BLACK, pygame.Rect(loc, (cs,cs)), thinkness)
        if thinkness == 2:
            for cell in range(1,10):
                self.display_zone(cell, cs//3, loc, 1)
        
    def display_entry(self, cs, pos, num, entryIndex):
        x = pos[0]
        y = pos[1]
        if self.sudoku_grid[x][y][0] != self.solution_grid[x][y][0] and self.check_solution:
            # wrong answer
            display_num = self.font.render(str(num), 1, RED)
            self.sudoku_has_error = True
        else:
            if entryIndex == 1: # New
                display_num = self.font.render(str(num), 1, BLUE)
            else: # Givem
                display_num = self.sysFont.render(str(num), 1, BLACK)
        SCREEN.blit(display_num, (self.margin + (x+1)*cs - 2*cs//3, self.margin + (y+1)*cs - 3*cs//4))
        
    def display_sudoku(self, s):
        SCREEN.fill((0,0,0))
        pygame.draw.rect(SCREEN, WHITE, pygame.Rect((self.margin, self.margin), (WIDTH - 2*self.margin, HEIGHT - 2*self.margin)))
        if self.byos:
            SCREEN.blit(self.sudoku_box_image, (50,50))
            pygame.draw.rect(SCREEN, WHITE, pygame.Rect((50,50), 
                                                        (self.sudoku_box_image.get_width(),
                                                         self.sudoku_box_image.get_height())), 2)
        else:
            for z in range(1,10):
                self.display_zone(z, self.cell_size, (self.margin, self.margin))
        
        if self.selected_cell_rect is not None:
            pygame.draw.rect(SCREEN, BLUE, self.selected_cell_rect, 1)

        for x in range(9):
            for y in range(9):
                if s[x][y][1] != 0: #User entered
                    self.display_entry(self.cell_size//3, (x,y), s[x][y][0], s[x][y][1])
                elif s[x][y][0] != 0 and not self.byos:
                    self.display_entry(self.cell_size//3, (x,y), s[x][y][0], 0)
                    
        if self.start_time != 0 and not self.solved and not self.sudoku_has_error and not self.sudoku_has_no_solution:
                self.time_taken = time.time() - self.start_time
                SCREEN.blit(self.font2.render("Lapsed " + str(int(self.time_taken)) +"s", 1, WHITE), (self.margin,10))  
    
    def show_image(self, title, img):
        cv.imshow(title, img) 
        cv.waitKey(0) 
        cv.destroyAllWindows()  
    
    def print_sudoku(self, s):
        for r in range(0,9,1):
            for c in range(0,9,1):
                if c==0 or c==3 or c==6:
                    print("  ",end="")
                if s[c][r][1] == 1:
                    print(Fore.LIGHTMAGENTA_EX + str(s[c][r][0]) + " ", end="")
                elif s[c][r][1] == 2:
                    print(Fore.RED + str(s[c][r][0]) + " ", end="")
                elif s[c][r][0] == 0:
                    print(Fore.LIGHTBLACK_EX + "- ", end="")
                else:
                    print(Fore.LIGHTGREEN_EX + str(s[c][r][0]) + " ", end="")
                if c==8:
                    print(" ",end="")
            print("")
            if r==5 or r == 2:
                print("")
            
#############################################################
if __name__ == "__main__":
    os.environ['SDL_VIDEO_CENTERED'] = '1'  # center SCREEN
    pygame.init()
    pygame.display.set_caption("Sudoku")
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    Runit = Sudoku()
    Myclock = pygame.time.Clock()
    while 1:
        if systemExit==True:
            pygame.quit()
            break;
        Runit.main()
        Myclock.tick(64)    
    exit()