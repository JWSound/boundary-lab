// Hidden Electron test of the actual SVG pointer-capture interaction.
const { app, BrowserWindow } = require("electron");
const { build } = require("esbuild");
const assert = require("node:assert/strict");
const { join } = require("node:path");

app.whenReady().then(async () => {
  setTimeout(() => { console.error("Crosshair test timed out"); app.exit(1); }, 45000).unref();
  const win = new BrowserWindow({ show: false, width: 1000, height: 700, webPreferences: { backgroundThrottling: false } });
  try {
    const bundle = await build({
      absWorkingDir: join(__dirname, ".."), bundle: true, write: false, format: "iife", jsx: "automatic",
      stdin: { resolveDir: join(__dirname, ".."), loader: "tsx", contents: `
        import React from 'react';
        import {createRoot} from 'react-dom/client';
        import {ElectricalPlot} from './src/components/ElectricalPlot';
        import {DriverExcursionPlot} from './src/components/DriverExcursionPlot';
        import './src/styles.css';
        const root=createRoot(document.getElementById('root'));
        const values=new Float32Array([1,2,3]);
        const data={key:'test',frequenciesHz:new Float64Array([20,100,1000]),traces:new Map([['driver',{
          name:'Driver',excursionMm:values,impedanceMagnitudeOhm:values,impedancePhaseDeg:values,
          rmsCurrentA:values,realPowerW:values,acousticResistance:values,acousticReactance:values,differentialPressurePa:values
        }]])};
        window.renderPlot=(view)=>root.render(React.createElement(view==='excursion'?DriverExcursionPlot:ElectricalPlot,{
          data,view,coupledSelected:true,currentFrequencyHz:100,frequencyPosition:1,frequencyCount:3,
          onFrequencyPositionChange:()=>{},canCalculate:true,calculating:false,completedCount:3,totalCount:3,onCalculateOrStop:()=>{}
        }));
      ` }, outdir: "unused-test-output",
    });
    const css = bundle.outputFiles.find((file) => file.path.endsWith(".css")).text;
    const js = bundle.outputFiles.find((file) => file.path.endsWith(".js")).text;
    await win.loadURL("data:text/html," + encodeURIComponent('<div id="root" style="width:900px;height:500px"></div><style>' + css + '.microphone-response{height:480px}</style>'));
    await win.webContents.executeJavaScript(js);
    const frame = () => win.webContents.executeJavaScript('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');
    for (const view of ["excursion", "impedance", "current", "power", "acoustic", "differential"]) {
      await win.webContents.executeJavaScript(`window.renderPlot('${view}')`);
      await frame();
      const point = await win.webContents.executeJavaScript(`(()=>{
        const well=document.querySelector('.plot-well'), r=well.getBoundingClientRect();
        if(getComputedStyle(well).fill!=='rgb(255, 255, 255)') throw Error('Not white');
        if(getComputedStyle(document.querySelector('.plot-grid.major')).stroke!=='rgb(128, 128, 128)') throw Error('Grid colour');
        return {x:Math.round(r.x+r.width*.4), y:Math.round(r.y+r.height*.4)};
      })()`);
      win.webContents.sendInputEvent({ type: "mouseDown", ...point, button: "left", clickCount: 1 });
      win.webContents.sendInputEvent({ type: "mouseMove", x: point.x + 15, y: point.y + 10 });
      win.webContents.sendInputEvent({ type: "mouseUp", x: point.x + 15, y: point.y + 10, button: "left", clickCount: 1 });
      await frame();
      assert.equal(await win.webContents.executeJavaScript("document.querySelectorAll('.plot-crosshair').length"), 2, view);
      assert.equal(await win.webContents.executeJavaScript("document.querySelectorAll('.crosshair-labels text').length"), view === "impedance" ? 3 : 2, view);
      await win.webContents.executeJavaScript("document.querySelector('svg.response-chart').dispatchEvent(new MouseEvent('dblclick',{bubbles:true}))");
      await frame();
      assert.equal(await win.webContents.executeJavaScript("document.querySelectorAll('.plot-crosshair').length"), 0, view);
    }
    console.log("White/grid styling and drag/release/clear crosshair passed for all six speaker views.");
    app.exit(0);
  } catch (error) { console.error(error); app.exit(1); }
});
