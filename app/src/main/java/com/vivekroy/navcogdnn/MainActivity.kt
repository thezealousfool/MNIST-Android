package com.vivekroy.navcogdnn

import android.graphics.drawable.BitmapDrawable
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var classifier: Classifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        classifier = Classifier(this)
    }

    public fun imgClicked(v : View) {
//        Log.d("vvk:", v.context.resources.getResourceName(v.id))
        val selectedImg = findViewById<ImageView>(R.id.selImg)
        val img = ((v as ImageView).drawable as BitmapDrawable).bitmap
        selectedImg.setImageBitmap(img)
        val result = classifier.classify(img)
        findViewById<TextView>(R.id.time).text = "${result.timeCost} ms"
        findViewById<TextView>(R.id.prob).text = "%.6f".format(result.probability)
        findViewById<TextView>(R.id.pred).text = "${result.number}"
    }
}