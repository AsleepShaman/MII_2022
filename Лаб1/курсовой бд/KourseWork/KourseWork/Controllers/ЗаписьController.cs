using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Web;
using System.Web.Mvc;
using KourseWork.Models;

namespace KoursWork.Controllers
{
    public class ЗаписьController : Controller
    {
        private AudioStudioDBEntities2 db = new AudioStudioDBEntities2();

        // GET: Запись
        public ActionResult Index()
        {
            var запись = db.Запись.Include(з => з.Перечень_услуг);
            return View(запись.ToList());
        }

        // GET: Запись/Details/5
        public ActionResult Details(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Запись запись = db.Запись.Find(id);
            if (запись == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(запись);
        }

        // GET: Запись/Create
        public ActionResult Create()
        {
            ViewBag.Код_услуги = new SelectList(db.Перечень_услуг, "Код_услуги", "Оборудование");
            return View();
        }

        // POST: Запись/Create
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Create([Bind(Include = "Код_услуги,Перезаписи,Фрагменты,Что_записывается,Сопр__инструменты,Битрейт")] Запись запись)
        {
            if (ModelState.IsValid)
            {
                try{db.Запись.Add(запись);
                db.SaveChanges();
                return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Ошибка создания";
                    return RedirectToAction("Error");
                }
            }

            ViewBag.Код_услуги = new SelectList(db.Перечень_услуг, "Код_услуги", "Оборудование", запись.Код_услуги);
            return View(запись);
        }

        // GET: Запись/Edit/5
        public ActionResult Edit(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Запись запись = db.Запись.Find(id);
            if (запись == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            ViewBag.Код_услуги = new SelectList(db.Перечень_услуг, "Код_услуги", "Оборудование", запись.Код_услуги);
            return View(запись);
        }

        // POST: Запись/Edit/5
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Edit([Bind(Include = "Код_услуги,Перезаписи,Фрагменты,Что_записывается,Сопр__инструменты,Битрейт")] Запись запись)
        {
            if (ModelState.IsValid)
            {
                try{db.Entry(запись).State = EntityState.Modified;
                db.SaveChanges();
                return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Ошибка редактировани!";
                    return RedirectToAction("Error");
                }
            }
            ViewBag.Код_услуги = new SelectList(db.Перечень_услуг, "Код_услуги", "Оборудование", запись.Код_услуги);
            return View(запись);
        }

        // GET: Запись/Delete/5
        public ActionResult Delete(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Запись запись = db.Запись.Find(id);
            if (запись == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(запись);
        }

        // POST: Запись/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public ActionResult DeleteConfirmed(string id)
        {
            try{Запись запись = db.Запись.Find(id);
            db.Запись.Remove(запись);
            db.SaveChanges();
            return RedirectToAction("Index");
            }
            catch
            {
                ViewBag.Data = "Ошибка удаления! Возможно, эти данные где-то есть...";
                return RedirectToAction("Error");
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                db.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
