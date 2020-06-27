from flask import Flask,render_template,redirect, request, send_from_directory, jsonify
from PIL import Image  
import pytesseract,json,cv2,base64,re
import App 

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='/static')

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)

@app.route('/uploads/<path:path>')
def send_uploads(path):
    return send_from_directory('png', path)
 

@app.route("/result", methods=["GET", "POST"])
def upload_image():
    global v_type,v_number
    if request.method == 'POST':
        uploadImg = request.files["fileUpload"]
        
        uploadImg.save("uploads/img.png") 
        input_im = cv2.imread("uploads/img.png")
        object_vals = App.objectDetection(input_im)
        v_type = object_vals[0]
        
        if object_vals[0] == 'car' or object_vals[0] == 'truck' or object_vals[0] =='bus':
            result_vals = App.Detect_number(object_vals[1])
            print(result_vals[0])
            
            if  result_vals[0] != '':
                v_number  = re.sub('[^A-Za-z0-9]+', '', result_vals[0]) 
                
            elif result_vals[0] != '000000000':
                second_check = App.casecade(object_vals[1])
                v_number  = re.sub('[^A-Za-z0-9]+', '', second_check[0]) 
            else:
                second_check = App.casecade(object_vals[1])
                v_number  = re.sub('[^A-Za-z0-9]+', '', second_check[0]) 
                #print(v_number)   
        else:
            v_number = "Did not Detected"
            
        
        return redirect(request.url)
    return jsonify(
        vechileType=v_type,
        vechileNumber=v_number
    )
    

@app.route('/')
def index():
    return render_template("index.html",)


if __name__ == '__main__':
    v_type = ''
    v_number = ''
    app.run(debug=True, port=8000, host='127.0.0.1')
    

