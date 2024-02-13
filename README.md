# Solve any Sudoku in seconds
Sudoku, the popular number puzzle game, is not just a test of logic but also a great exercise in problem-solving for programmers. As someone deeply passionate about software development and AI, I took on the challenge of creating a program that could solve any Sudoku puzzle. 

## Scanning Sudoku using Tesseract / OCR
Tesseract is an OCR engine that uses LSTM-based neural networks to recognize character patterns. Pytesseract is excellent at reading printed text, which means there is no need to build custom models unless you are going for scanning hand-crafted Sudokus 

```
text = pytesseract.image_to_string(img, lang="eng",config='--psm 6 --oem 3')
```

## Solve Skewing Problem using OpenCV
It's easy to implement a four-point perspective transform using OpenCV and correct the skewed grid into a perfect square. This transformation greatly improves the OCR's accuracy

<img src="https://static.wixstatic.com/media/9c8449_715f7c6b14ee4844a960392c500481f3~mv2.png/v1/fill/w_700,h_658,al_c,q_90,usm_0.66_1.00_0.01,enc_auto/9c8449_715f7c6b14ee4844a960392c500481f3~mv2.png"
     alt="Sudoko Grid" width="200"
     style="float: left; margin-right: 10px;" />

```
def four_point_transform(self, image, rect, dst, width, height):
      M = cv.getPerspectiveTransform(rect, dst)
      warped = cv.warpPerspective(image, M, (width, height))
      return warped
```

## Backpropagation to solve Sudoku 
Turns out there are many ways to automatically solve a Sudoku. I like the back-propagation algorithm as it is simple and works every time. It also makes the animation on Pygame very interesting. 


