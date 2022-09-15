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
    public class АранжировкаController : Controller
    {
        private AudioStudioDBEntities2 db = new AudioStudioDBEntities2();

        // GET: Аранжировка
        public ActionResult Index()
        {
            var аранжировка = db.Аранжировка.Include(а => а.Перечень_услуг);
            return View(аранжировка.ToList());
        }

        // GET: Аранжировка/Details/5
        public ActionResult Details(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Аранжировка аранжировка = db.Аранжировка.Find(id);
            if (аранжировка == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(аранжировка);
        }

        // GET: Аранжировка/Create
        public ActionResult Create()
        {
            ViewBag.Код_услуги = new SelectList(db.Перечень_услуг, "Код_услуги", "Оборудование");
            return View();
        }

        // POST: Аранжировка/Create
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Create([Bind(Include = "Код_услуги,Референс,Жанр,Кол_во_музыкантов")] Аранжировка аранжировка)
        {
            if (ModelState.IsValid)
            {
                try
                {
                    db.Аранжировка.Add(аранжировка);
                    db.SaveChanges();
                    return RedirectToAction("Index");
                }
                    catch
                {
                    ViewBag.Data = "Ошибка создания записи!";
                    return RedirectToAction("Error");
                }
            }

            ViewBag.Код_услуги = new SelectList(db.Перечень_услуг, "Код_услуги", "Оборудование", аранжировка.Код_услуги);
            return View(аранжировка);
        }

        // GET: Аранжировка/Edit/5
        public ActionResult Edit(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Аранжировка аранжировка = db.Аранжировка.Find(id);
            if (аранжировка == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            ViewBag.Код_услуги = new SelectList(db.Перечень_услуг, "Код_услуги", "Оборудование", аранжировка.Код_услуги);
            return View(аранжировка);
        }

        // POST: Аранжировка/Edit/5
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Edit([Bind(Include = "Код_услуги,Референс,Жанр,Кол_во_музыкантов")] Аранжировка аранжировка)
        {
            if (ModelState.IsValid)
            {
                try
                {
                    db.Entry(аранжировка).State = EntityState.Modified;
                    db.SaveChanges();
                    return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Ошибка редактирования!";
                    return RedirectToAction("Error");
                }
            }
            ViewBag.Код_услуги = new SelectList(db.Перечень_услуг, "Код_услуги", "Оборудование", аранжировка.Код_услуги);
            return View(аранжировка);
        }

        // GET: Аранжировка/Delete/5
        public ActionResult Delete(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Аранжировка аранжировка = db.Аранжировка.Find(id);
            if (аранжировка == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(аранжировка);
        }

        // POST: Аранжировка/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public ActionResult DeleteConfirmed(string id)
        {
            try
            {
                Аранжировка аранжировка = db.Аранжировка.Find(id);
                db.Аранжировка.Remove(аранжировка);
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
