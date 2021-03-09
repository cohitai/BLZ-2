# BLZ-2

Build instructions:

A. Clone the Repo. 
B. Docker Build
C. Lauching commands: python main.py -M 500 20 10 4 -A 30 OR Python main.py -A 30
   parameters meaning: -M x y z t to fit a new model on start. 
                        :params embedding size, window size, minimum word occurence, numbers of workers
                        -A to start automation. :param number of day for relevant articles. The default is 30 days. 
                        optional flags: -D <Server name> . Changes DNS.  The default is apiblzapp.tk 
                        -S To set the inner struture of directories, relevant only for the first run.
                        -R,V visualizations to produce graphs. Do not work since it wasn't updated since the first version. 
                        
                        -P prediction. For testing purposes, generates one prediction and quits. 
                        
                        
                        
4. In the first build use the docker file with "Docker build" and replace the last CMD by the following: 


CMD python.py -S -M 500 20 10 4 -A 50 

5. Push the Image to GCP Container repos. (Docker tag, docker push). Run with GKE using the deployments protoypes with "kubectl apply -f <Deployment>"
or, run on a seperate VM with Docker run <image>.



