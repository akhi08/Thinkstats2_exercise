

🛠️ How to Use It
-

Create a file called run_all.sh in your project root:
-
- touch run_all.sh

Edit the file to paste the script above:
- 
- nano run_all.sh
- when terminal opens,paste: 
- - #!/bin/sh
- - - python train.py --fold 0
- - - python train.py --fold 1
- - - python train.py --fold 2
- - - python train.py --fold 3
- - - python train.py --fold 4
    - close by Ctrl+O -> ENTER ->Ctrl+X


Make it executable:
-
- chmod +x run_all.sh

Run the script:
-
- ./run_all.sh

<!--  --> you can find the path  by running :
<!--  -->find . -name train.py
