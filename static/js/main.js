// ====================== thumbnail preview =========================
function readImage(input) {
    // 인풋 태그에 파일이 있는 경우
    if (input.files && input.files[0]) {

        // FileReader 인스턴스 생성
        const reader = new FileReader()

        // 이미지가 로드된 경우
        reader.onload = e => {
            const previewImage = document.getElementById("preview-image")
            previewImage.src = e.target.result
        }
        // reader가 이미지 읽도록 하기
        reader.readAsDataURL(input.files[0])
    }
}

// input file에 change 이벤트 부여
const inputImage = document.getElementById("input-image")
inputImage.addEventListener("change", e => {
    readImage(e.target)
})

// =========================== image upload ===========================

window.onload = function () {
    const userImg = document.querySelector(".user-img")
    const uploader = document.querySelector(".uploder")
    uploader.addEventListener("click", upload)
}


function upload(e) {
    var formData = new FormData(document.getElementById("upload-form"));
    formData.append("img", $("#input-image")[0].files[0]);

    $.ajax({
        url: "/",
        contentType: false,
        processData: false,
        type: "POST",
        data: formData,
        success: function (response) {

            const equationDiv = document.querySelector("#equation");
            equationDiv.innerHTML = response["result"];
            MathJax.Hub.Queue(["Typeset", MathJax.Hub]); // 이거 써야 latex 형태로 보임

        },
        error: function (response) {
            alert("upload failed.")
        },
    })

}
