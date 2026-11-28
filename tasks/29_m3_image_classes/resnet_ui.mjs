// 本物の ResNet18 の既製のモデルの一式を ML-Provide の画面から作って登録し,
// 画像のディレクトリを推論して種類の名前を読むブラウザの操作.
// resnet_ui.py が引数 (JSON) を渡して動かす. Playwright は ML-Provide の画面のものを読む
import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const task = JSON.parse(process.argv[2]);
const require = createRequire(path.join(task.frontend, "package.json"));
const { chromium, expect } = require("@playwright/test");

// 段階の記録 (resnet_ui.py が表に写す)
const steps = [];
const labels = {};
let error = null;
const add = (step, status, note) => steps.push({ step, status, note });

// 重みを出どころから読んでハッシュ値を求める間と, 推論を待つ時間
const LONG = 600_000;

fs.mkdirSync(task.shots, { recursive: true });
let browser = null;
let page = null;
try {
  browser = await chromium.launch();
  page = await browser.newPage({
    baseURL: task.base,
    viewport: { width: 1600, height: 900 },
  });
  // 起動ごとのトークンを Cookie に保存して画面を開く (利用者が開く URL と同じ)
  await page.goto(`/api/auth?token=${encodeURIComponent(task.token)}`);
  await expect(
    page.getByRole("button", { name: "既製のモデルから作る" }),
  ).toBeVisible();
  add(
    "画面を開く (トークンの URL)",
    "OK",
    "一式の一覧に「既製のモデルから作る」",
  );

  // 出どころ, モデル, 重みを選んで下見する
  await page.getByRole("button", { name: "既製のモデルから作る" }).click();
  const dialog = page.getByRole("dialog", { name: "既製のモデルから作る" });
  await dialog.getByLabel("出どころ").selectOption(task.library);
  await dialog.getByLabel("モデルの名前").fill(task.model);
  await dialog.getByLabel("重みの名前 (版)").fill(task.weights);
  const preview = dialog.getByRole("region", { name: "下見" });
  await expect(preview).toContainText("1,000 種類", { timeout: 60_000 });
  await page.screenshot({ path: path.join(task.shots, "1_preview.png") });
  const seen = (await preview.innerText()).replace(/\s+/g, " ");
  add("既製のモデルの下見", "OK", seen.slice(0, 160));

  // 作って登録する (重みを出どころから読んでハッシュ値を求める)
  await dialog.getByLabel("一式の名前").fill(task.name);
  await dialog.getByRole("button", { name: "作って登録" }).click();
  await expect(page.getByRole("heading", { name: task.name })).toBeVisible({
    timeout: LONG,
  });
  const store = page.getByText(/置き場にあります|置き場にありません/);
  await expect(store).toBeVisible();
  await expect(
    page.getByText(
      new RegExp(`${task.library} の ${task.model} \\(重み ${task.weights}\\)`),
    ),
  ).toBeVisible();
  await page.screenshot({
    path: path.join(task.shots, "2_bundle.png"),
    fullPage: true,
  });
  add(
    "作って登録 (一式の画面)",
    "OK",
    `出どころ ${task.library} の ${task.model} (重み ${task.weights}). ${(await store.innerText()).split(" (")[0]}`,
  );

  // 画像のディレクトリを推論する
  await page.getByLabel(/推論する画像/).fill(task.images);
  await page.getByRole("button", { name: "推論する" }).click();
  const summary = page.getByText(
    `${task.count} 件のうち ${task.count} 件を予測`,
  );
  await expect(summary).toBeVisible({ timeout: LONG });
  await page.screenshot({
    path: path.join(task.shots, "3_result.png"),
    fullPage: true,
  });
  add(
    "画像のディレクトリを推論",
    "OK",
    `${task.count} 件のうち ${task.count} 件を予測`,
  );

  // 予測と判断の表から, 画像ごとの種類の名前 (prediction の列) を読む
  const table = page.getByRole("table", { name: "予測と判断の表" });
  const headers = (await table.getByRole("columnheader").allInnerTexts()).map(
    (t) => t.trim(),
  );
  const column = headers.indexOf("prediction");
  for (const row of await table.getByRole("row").all()) {
    const cells = (await row.getByRole("cell").allInnerTexts()).map((t) =>
      t.trim(),
    );
    const file = cells.find((c) => /\.png$/.test(c));
    if (file && column >= 0) labels[path.basename(file)] = cells[column];
  }
  const read = Object.keys(labels).length;
  add(
    "種類の名前が表に出る",
    read === task.count ? "OK" : "誤り",
    `列 ${headers.join(", ")}. ${read} 件の種類の名前`,
  );
} catch (caught) {
  error = String(caught?.message ?? caught).split("\n")[0];
  add("止まった画面の操作", "誤り", error);
  await page
    ?.screenshot({ path: path.join(task.shots, "error.png"), fullPage: true })
    .catch(() => {});
} finally {
  await browser?.close().catch(() => {});
  fs.writeFileSync(
    task.result,
    JSON.stringify({ steps, labels, error }, null, 1),
  );
}
process.exit(error ? 1 : 0);
