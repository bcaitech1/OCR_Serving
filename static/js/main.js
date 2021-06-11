window.onload = function () {
    const userImg = document.querySelector(".user-img")
    const uploader = document.querySelector(".uploder")
    uploader.addEventListener("click", upload)
}

// input file에 change 이벤트 부여
const inputImage = document.getElementById("input-image")
inputImage.addEventListener("change", e => {
    readImage(e.target)
})

const equationDiv = document.getElementById("equation")
const equationLabel = document.getElementById("equation-label")
const latexDiv = document.getElementById("latex")
const latexLabel = document.getElementById("latex-label")
const previewImage = document.getElementById("preview-image")



// ====================== thumbnail preview =========================
function readImage(input) {
    // 인풋 태그에 파일이 있는 경우
    if (input.files && input.files[0]) {

        // FileReader 인스턴스 생성
        const reader = new FileReader()

        // 이미지가 로드된 경우
        reader.onload = e => {
            previewImage.src = e.target.result

            equationLabel.innerText = " "
            equationDiv.innerText = " "
            latexLabel.innerText = " "
            latexDiv.innerText = " "
            // 새 이미지를 로드하면 equation 지우기

        }
        // reader가 이미지 읽도록 하기
        reader.readAsDataURL(input.files[0])
    }
}


// =========================== image upload ===========================
function upload(e) {
    var formData = new FormData(document.getElementById("upload-form"))
    formData.append("img", $("#input-image")[0].files[0])

    $.ajax({
        url: "/",
        contentType: false,
        processData: false,
        type: "POST",
        data: formData,
        beforeSend: function () {
            displayLoadingbar()
        },
        complete: function () {
            hideLoadingbar()
        },
        success: function (response) {
            const result = response["result"]
            const result_len = result.length

            equationLabel.innerText = "Recognized equation"
            equationDiv.innerText = result

            latexLabel.innerText = "Pasteable LaTeX"
            latexDiv.innerText = result.substring(2, result_len-2) // 앞뒤 두글자씩 자르기

            MathJax.Hub.Queue(["Typeset", MathJax.Hub]) // LaTeX 렌더링을 다시 요청?
        },
        error: function (response) {
            alert("upload failed.")
        },
    })

}

// =========================== loading bar ===========================
function displayLoadingbar() {
    const backHeight = $(document).height() //뒷 배경의 상하 폭
    const backWidth = window.document.body.clientWidth //뒷 배경의 좌우 폭
    const backGroundCover = "<div id='back'></div>" //뒷 배경을 감쌀 커버

    let loadingBarImage = '' //가운데 띄워 줄 이미지
    loadingBarImage += "<div id='loadingBar'>"
    loadingBarImage += " <img src='/static/img/loadingbar.gif'/>" //로딩 바 이미지
    loadingBarImage += "</div>"

    $('body').append(backGroundCover).append(loadingBarImage)
    $('#back').css({'width': backWidth, 'height': backHeight, 'opacity': '0.3'}).show()
    $('#loadingBar').show()
}

function hideLoadingbar() {
    $('#back, #loadingBar').hide()
    $('#back, #loadingBar').remove()
}
