# BLZ-2

Build instructions:

1. Clone the Repo. 
2. Docker Build
3. lauching commands: (python main.py -M 500 20 10 4 -A 30)
   parameters meaning: -M x y z t to fit a new model on start. :params embedding size, window size, minimum word occurence, numbers of workers
                        -A to start automation. :param number of day for relevant articles. The default is 30 days. 
                        optional flags: -D <Server name> . The default is apiblzapp.tk 
                        -S to set the inner struture of directory, relevant only for the first run.
                        -R,V visualization to produce graphs. Do not work since it wasn't updated from the first version. 
                        -P prediction. For testing purposes, generates one prediction and quits. 
                        
                        
                        
4. For first build use the docker file with "Docker build" and replace the last CMD with: 


CMD python.py -S -M 500 20 10 4 -A 50 

5. Push the Image to GCP Container repos. and run with either Kubernetes with the deployments protoype with "kubectl apply -f <Deployment>"
or, run on a seperate VM with Docker run <image>.



