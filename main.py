import datetime
import os
import sys
from datetime import datetime
import cv2
import face_recognition
import pandas
import pyttsx3
import speech_recognition as sr
import wikipedia
import wolframalpha


# pip install pipwin
# pipwin install pyaudio


def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening...', end=" ")
        r.pause_threshold = 0.7
        audio = r.listen(source)
        try:
            print("Recognizing...")
            Query = r.recognize_google(audio, language='en-in')
            print("Recognized as: ", Query)
        except Exception as e:
            print(e)
            print("Please be more clear")
            return "none"
        return Query


def speak(audio):
    strlen = len(audio.split())
    if strlen >= 20:
        audio = audio.split()[:20]
        audio = ' '.join(map(str, audio))
    print(audio)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.say(audio)
    engine.runAndWait()


def talking_tom():
    while True:
        talk = takeCommand().lower()
        if "none" in talk:
            speak("")
        if "exit" in talk:
            speak("Thank you for spending your time with me.")
            bin()
            break
        speak(talk)


def tellDay():
    day = datetime.today().weekday() + 1
    Day_dict = {1: 'Monday', 2: 'Tuesday',
                3: 'Wednesday', 4: 'Thursday',
                5: 'Friday', 6: 'Saturday',
                7: 'Sunday'}
    if day in Day_dict.keys():
        day_of_the_week = Day_dict[day]
        speak("The day is " + day_of_the_week)
    bin()


def tellTime():
    time = str(datetime.now())
    hour = time[11:13]
    min1 = time[14:16]
    speak("The time is sir" + hour + "Hours and" + min1 + "Minutes")
    bin()


def Hello():
    speak("hello sir I am your desktop assistant.Tell me how may I help you")


def bin():
    while True:
        query = takeCommand().lower()
        if query == "none":
            continue
        elif "hello" in query:
            Hello()
            continue
        elif "bhai" in query:
            speak("Thank you i will take a leave")
            Take_query()
            break
        elif "bye" in query:
            speak("Thank you i will take a leave")
            Take_query()
            break
        elif "from wikipedia" in query:
            speak("Checking the wikipedia ")
            query = query.replace("wikipedia", "")
            result = wikipedia.summary(query, sentences=1)
            speak("According to wikipedia")
            speak(result)
        elif "talking tom" in query:
            speak("Loading Talking Tom please wait")
            talking_tom()
            break
        elif "day" in query:
            tellDay()
            break
        elif "time" in query:
            tellTime()
            break
        else:
            try:
                res = client.query(query)
                output = next(res.results).text
                speak(output)
            except:
                speak("Please be more clear")


def most_frequent(List):
    return max(set(List), key=List.count)


def markAttendance(std_name):
    with open('C:/Users/shibi/Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if std_name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{std_name},{dtString}')


def check(name_list):
    if len(name_list) % 100 == 0 and most_frequent(name_list) != "Unknown":
        std_name = most_frequent(name_list)
        speak('detected as' + std_name)
        while True:
            check = takeCommand().lower()
            if check == "yes":
                markAttendance(std_name)
                speak("Thank You Next Person please")
                name_list = []
            elif check == 'quit':
                name_list = []
                speak('Exiting Attendance system')
                cv2.destroyAllWindows()
                Take_query()
            elif check == "no":
                speak("Please try again")
                name_list = []
            if check == "yes" or check == "no" or check == "quit":
                break


path = 'E:/Projects/Python/Face-Recogntion/photos'
images = []
classNames = []
allList = []
regNo = []
myList = os.listdir(path)
for name in myList:
    curImg = cv2.imread(f'{path}/{name}')
    images.append(curImg)
    all = name
    name = name.split('_')
    reg = name[0]
    name = name[1]
    regNo.append(os.path.splitext(reg)[0])
    classNames.append(os.path.splitext(name)[0])
    allList.append(os.path.splitext(all)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
speak("Completed Encoding")

known_face_encodings = encodeListKnown
known_face_names = classNames
speak(known_face_names)


def face_rec():
    # Initialize some variables
    video_capture = cv2.VideoCapture(0)
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    # Csv file
    df = pandas.DataFrame(columns=["Start", "End"])
    count = "Unknown"
    name_list = []

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        status = 0
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            # face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=0, model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                # name check
                face_names.append(name)
                if count != name:
                    count = name
                    # speak(name)                            #--> To speak the detected name and the change in name

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            status = 1
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 1)

            # Draw a label with a name below the face
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            name_list.append(name)

            # Check Function to check whether the person is right and Append to CSV
            check(name_list)
        # Display the resulting image
        cv2.imshow("Video", frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


client = wolframalpha.Client('7E2K87-L6AH35EQ44')


def Take_query():
    while True:
        query = takeCommand().lower()
        if "bin" in query:
            speak("Hello i am bin bot, here to help you")
            bin()
        if "attendance" in query:
            speak("Loading attendance")
            face_rec()
        if "exit" in query:
            speak("Turning off Bin bot, Thankyou")
            sys.exit()
            break


Take_query()
